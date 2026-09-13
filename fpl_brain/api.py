"""Transport-only client for the public Fantasy Premier League API."""

from __future__ import annotations

import json
import logging
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal

import requests

from .utils import ensure_directory

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://fantasy.premierleague.com/api/"


class FplApiError(Exception):
    """Base class for transport and response failures."""


class FplHttpError(FplApiError):
    def __init__(self, message: str, status_code: int, endpoint: str):
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint


class FplNotFoundError(FplHttpError):
    """A 404, which is expected for pre-deadline picks."""


class FplTimeoutError(FplApiError):
    """The endpoint could not be reached before the configured timeout."""


class FplInvalidResponseError(FplApiError):
    """The endpoint returned invalid JSON or an unexpected top-level shape."""


class FplClient:
    """Small synchronous client with retries and raw response persistence."""

    def __init__(
        self,
        config: dict[str, Any],
        raw_dir: str | Path | None = None,
        run_id: int | str | None = None,
        dry_run: bool = False,
    ) -> None:
        request_config = config.get("request", {})
        self.timeout = (
            float(request_config.get("timeout_connect_seconds", 5)),
            float(request_config.get("timeout_read_seconds", 20)),
        )
        self.max_retries = int(request_config.get("max_retries", 3))
        self.backoff_base = float(request_config.get("backoff_base_seconds", 1.5))
        self.polite_delay = float(request_config.get("polite_delay_seconds", 0.4))
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None
        self.run_id = str(run_id) if run_id is not None else None
        self.dry_run = dry_run
        self._last_request_at: float | None = None
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": request_config.get("user_agent", "fpl-brain/1.0")})

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "FplClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def set_raw_run(self, run_id: int | str | None) -> None:
        self.run_id = str(run_id) if run_id is not None else None

    def _raw_path(self, endpoint_slug: str) -> Path | None:
        if self.dry_run or self.raw_dir is None or self.run_id is None:
            return None
        safe_slug = "".join(char if char.isalnum() or char in "-_" else "_" for char in endpoint_slug)
        directory = ensure_directory(self.raw_dir / self.run_id)
        return directory / f"{safe_slug}.json"

    @staticmethod
    def _retry_after(response: requests.Response | None) -> float | None:
        if response is None:
            return None
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                return max(0.0, date.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None

    def _sleep_before_retry(self, attempt: int, response: requests.Response | None = None) -> None:
        retry_after = self._retry_after(response)
        delay = retry_after if retry_after is not None else self.backoff_base**attempt
        time.sleep(max(0.0, delay) + random.uniform(0.0, 0.1))

    def _get(
        self,
        path: str,
        expect: Literal["dict", "list"],
        endpoint_slug: str | None = None,
        bootstrap_shape: bool = False,
    ) -> dict[str, Any] | list[Any]:
        endpoint = path
        slug = endpoint_slug or path.strip("/").replace("/", "_").replace("?", "_").replace("=", "_")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            # Apply the configured politeness interval to every outbound request,
            # including endpoint fan-out and retries.  The first request is not
            # delayed because there is no prior request to space from.
            self._polite_pause()
            started = time.monotonic()
            response: requests.Response | None = None
            try:
                response = self.session.get(BASE_URL + path.lstrip("/"), timeout=self.timeout)
                self._last_request_at = time.monotonic()
                elapsed = time.monotonic() - started
                LOGGER.info("request endpoint=%s status=%s elapsed=%.3fs", endpoint, response.status_code, elapsed)
                if response.status_code == 404:
                    raise FplNotFoundError(f"404 from {endpoint}", 404, endpoint)
                if response.status_code == 429 or response.status_code >= 500:
                    error = FplHttpError(f"HTTP {response.status_code} from {endpoint}", response.status_code, endpoint)
                    last_error = error
                    if attempt < self.max_retries:
                        self._sleep_before_retry(attempt, response)
                        continue
                    raise error
                if not 200 <= response.status_code < 300:
                    raise FplHttpError(f"HTTP {response.status_code} from {endpoint}", response.status_code, endpoint)

                raw_body = response.text
                raw_path = self._raw_path(slug)
                if raw_path is not None:
                    raw_path.write_text(raw_body, encoding="utf-8")
                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError as exc:
                    raise FplInvalidResponseError(f"Non-JSON response from {endpoint}") from exc
                if expect == "dict" and not isinstance(payload, dict):
                    raise FplInvalidResponseError(f"Expected an object from {endpoint}")
                if expect == "list" and not isinstance(payload, list):
                    raise FplInvalidResponseError(f"Expected an array from {endpoint}")
                if bootstrap_shape:
                    required = {"elements", "teams", "events"}
                    if not required.issubset(payload):  # type: ignore[arg-type]
                        missing = ", ".join(sorted(required.difference(payload)))  # type: ignore[arg-type]
                        raise FplInvalidResponseError(f"bootstrap-static missing top-level keys: {missing}")
                return payload
            except FplNotFoundError:
                raise
            except FplHttpError:
                raise
            except requests.Timeout as exc:
                self._last_request_at = time.monotonic()
                elapsed = time.monotonic() - started
                LOGGER.info("request endpoint=%s status=%s elapsed=%.3fs", endpoint, "timeout", elapsed)
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise FplTimeoutError(f"Timed out contacting {endpoint}") from exc
            except requests.RequestException as exc:
                self._last_request_at = time.monotonic()
                elapsed = time.monotonic() - started
                LOGGER.info("request endpoint=%s status=%s elapsed=%.3fs", endpoint, "error", elapsed)
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise FplApiError(f"Could not contact {endpoint}: {exc}") from exc
        raise FplApiError(f"Could not contact {endpoint}: {last_error}")

    def _polite_pause(self) -> None:
        if self._last_request_at is None or self.polite_delay <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.polite_delay:
            time.sleep(self.polite_delay - elapsed)

    def get_bootstrap_static(self) -> dict[str, Any]:
        return self._get("bootstrap-static/", "dict", "bootstrap_static", bootstrap_shape=True)  # type: ignore[return-value]

    def get_fixtures(self, event: int | None = None) -> list[Any]:
        path = "fixtures/" if event is None else f"fixtures/?event={int(event)}"
        slug = "fixtures" if event is None else f"fixtures_event_{int(event)}"
        return self._get(path, "list", slug)  # type: ignore[return-value]

    def get_element_summary(self, player_id: int) -> dict[str, Any]:
        return self._get(f"element-summary/{int(player_id)}/", "dict", f"element_summary_{int(player_id)}")  # type: ignore[return-value]

    def get_event_live(self, event: int) -> dict[str, Any]:
        return self._get(f"event/{int(event)}/live/", "dict", f"event_live_{int(event)}")  # type: ignore[return-value]

    def get_entry(self, entry_id: int) -> dict[str, Any]:
        return self._get(f"entry/{int(entry_id)}/", "dict", f"entry_{int(entry_id)}")  # type: ignore[return-value]

    def get_entry_history(self, entry_id: int) -> dict[str, Any]:
        return self._get(f"entry/{int(entry_id)}/history/", "dict", f"entry_history_{int(entry_id)}")  # type: ignore[return-value]

    def get_entry_transfers(self, entry_id: int) -> list[Any] | None:
        """Return the public transfer history, or None when the endpoint is unavailable."""

        try:
            return self._get(f"entry/{int(entry_id)}/transfers/", "list", f"entry_{int(entry_id)}_transfers")  # type: ignore[return-value]
        except FplNotFoundError:
            return None

    def get_entry_picks(self, entry_id: int, event: int) -> dict[str, Any] | None:
        try:
            return self._get(
                f"entry/{int(entry_id)}/event/{int(event)}/picks/",
                "dict",
                f"entry_{int(entry_id)}_event_{int(event)}_picks",
            )  # type: ignore[return-value]
        except FplNotFoundError:
            return None

    def get_event_status(self) -> dict[str, Any]:
        return self._get("event-status/", "dict", "event_status")  # type: ignore[return-value]
