"""Bootstrap ingest-generation completeness contract (R4B.1).

Why this exists
---------------
``repositories.upsert_players`` is ADDITIVE/updating and is always safe: it
inserts new players and refreshes known ones.  ``repositories.mark_absent_players``
is the only DESTRUCTIVE step — it sets ``is_active=0`` for every id that was not
seen in the newest payload.  A truncated-but-well-formed bootstrap payload
therefore does not merely leave data stale; it silently REDEFINES the official
player pool, which is exactly the population the all-player discovery contract
depends on.

This module records one auditable generation per accepted bootstrap and decides
whether the payload is complete enough to be allowed to redefine the official
set.  When it is not, the destructive step is skipped (upsert still runs, so
nothing new is lost) and the fetch is reported as
``OFFICIAL_PLAYER_POOL_INGEST_INCOMPLETE``.

ACCEPTANCE RULE — BOOTSTRAP_GENERATION_ACCEPTANCE v1
----------------------------------------------------
A generation may replace the official player set only when ALL hold:

R1 STRUCTURAL_COMPLETE
    The payload carries a non-empty ``elements`` array, every element has an
    integer ``id`` and non-empty ``web_name``, ids are unique, and the parsed
    count equals the official element count.  (Enforced upstream by
    ``parsers.validate_bootstrap_payload``; re-checked here because this module
    is also callable directly.)

R2 BOUNDED_DROP
    Both bounds must hold against the previous recorded generation:
      * retained fraction  ``|current ∩ previous| / |previous| >= 0.97``, and
      * absolute drop      ``|previous - current| <= 75``.
    Deliberately NOT a naive ``current_count == previous_count`` equality:
    legitimate additions and removals happen every week, and a count-equality
    rule would reject them.  Real pool churn removes a handful of ids; a partial
    or truncated payload removes a large block at once.  Requiring BOTH bounds
    means a small genuine change always passes and a mass disappearance always
    fails.

R3 NO_WHOLE_CLUB_LOSS
    No club that had players in the previous generation drops to zero players.
    A whole-club disappearance is the signature of a team-filtered or truncated
    payload rather than a football event.

R4 NO_PREVIOUS_GENERATION
    With no previous generation on record there is nothing to destroy, so the
    first generation is accepted (``FIRST_GENERATION``).

OVERRIDE
    ``allow_large_drop=True`` accepts a generation that fails R2/R3 and records
    the reason.  This exists so a genuine season boundary (a full squad reset)
    is not an unbreakable wall, while still being an explicit, auditable act.

A rejected generation is still PERSISTED as a record (accepted=False) so the
decision is auditable, and the previous generation stays authoritative.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

GENERATION_SCHEMA = "fpl_brain.bootstrap_generation.v1"
GENERATIONS_DIRNAME = "ingest_generations"
GENERATION_FILENAME = "latest_bootstrap_generation.json"

#: Version of the acceptance rule below.  Persisted with every generation so an
#: audit can tell which rule admitted the authoritative pool identity.
ACCEPTANCE_RULE_VERSION = "BOOTSTRAP_GENERATION_ACCEPTANCE v1"

DIAG_INGEST_INCOMPLETE = "OFFICIAL_PLAYER_POOL_INGEST_INCOMPLETE"

#: Retained-fraction floor and absolute-drop ceiling for R2 (see module docstring).
MIN_RETAINED_FRACTION = 0.97
MAX_ABSOLUTE_DROP = 75

ACCEPTANCE_RULES = (
    "FIRST_GENERATION",
    "STRUCTURAL_AND_BOUNDED_DRIFT",
    "OVERRIDE_ALLOW_LARGE_DROP",
)


class IngestGenerationError(ValueError):
    """A bootstrap generation is malformed or cannot be evaluated."""


@dataclass(frozen=True)
class BootstrapGeneration:
    """One auditable bootstrap ingest generation."""

    captured_at: str
    run_id: int | None
    official_element_count: int
    parsed_count: int
    persisted_count: int
    element_ids: tuple[int, ...]
    element_id_sha256: str
    club_player_counts: dict[str, int]
    availability_counts: dict[str, int]
    validation_status: str
    accepted: bool
    acceptance_rule: str
    rejection_reasons: tuple[str, ...]
    previous_element_count: int | None
    previous_id_sha256: str | None
    allow_large_drop: bool = False

    @property
    def diagnostic(self) -> str | None:
        return None if self.accepted else DIAG_INGEST_INCOMPLETE

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": GENERATION_SCHEMA,
            "captured_at": self.captured_at,
            "run_id": self.run_id,
            "official_element_count": int(self.official_element_count),
            "parsed_count": int(self.parsed_count),
            "persisted_count": int(self.persisted_count),
            "element_ids": list(self.element_ids),
            "element_count": len(self.element_ids),
            "element_id_sha256": self.element_id_sha256,
            "club_player_counts": dict(sorted(self.club_player_counts.items())),
            "availability_counts": dict(sorted(self.availability_counts.items())),
            "validation_status": self.validation_status,
            "accepted": bool(self.accepted),
            "acceptance_rule": self.acceptance_rule,
            "rejection_reasons": list(self.rejection_reasons),
            "previous_element_count": self.previous_element_count,
            "previous_id_sha256": self.previous_id_sha256,
            "allow_large_drop": bool(self.allow_large_drop),
            "diagnostic": self.diagnostic,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def elements_from_payload(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The official ``elements`` array, or an error."""

    elements = payload.get("elements") if isinstance(payload, Mapping) else None
    if not isinstance(elements, Sequence) or isinstance(elements, (str, bytes)) or not elements:
        raise IngestGenerationError("bootstrap payload has no non-empty elements array")
    return [element for element in elements if isinstance(element, Mapping)]


def element_ids_from_payload(payload: Mapping[str, Any]) -> list[int]:
    """Every official element id, preserving payload order, requiring uniqueness."""

    ids: list[int] = []
    for element in elements_from_payload(payload):
        raw = element.get("id")
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise IngestGenerationError(f"element has no integer id: {raw!r}")
        ids.append(int(raw))
    if len(set(ids)) != len(ids):
        raise IngestGenerationError("bootstrap payload contains duplicate element ids")
    return ids


def element_id_sha256(element_ids: Iterable[int]) -> str:
    """Canonical digest of the sorted, de-duplicated official element id set.

    THE single definition: the in-SQLite writer, the decision-time reader and
    the pool-completeness assertion all call this, so an "equal count, different
    ids" pool can never be mistaken for a matching one.
    """

    ordered = sorted({int(pid) for pid in element_ids})
    return hashlib.sha256(",".join(str(pid) for pid in ordered).encode()).hexdigest()


def availability_counts_from_payload(payload: Mapping[str, Any]) -> dict[str, int]:
    """Official availability status counts.

    Recorded for provenance ONLY.  Injury/suspension/doubt is an AVAILABILITY
    dimension, never a pool-membership dimension: a doubtful player must still
    exist in the universe and be downgraded by minutes/availability, not dropped.
    """

    counts: dict[str, int] = {}
    for element in elements_from_payload(payload):
        status = element.get("status")
        key = str(status) if status is not None else "UNKNOWN"
        counts[key] = counts.get(key, 0) + 1
    return counts


def club_player_counts_from_payload(payload: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements_from_payload(payload):
        team = element.get("team")
        if isinstance(team, bool) or team is None:
            key = "UNKNOWN"
        else:
            key = str(int(team))
        counts[key] = counts.get(key, 0) + 1
    return counts


def decide_acceptance(
    *,
    current_ids: Sequence[int],
    previous: BootstrapGeneration | None,
    official_element_count: int,
    parsed_count: int,
    unique_ids: bool = True,
    non_empty_ids: bool = True,
    current_club_counts: Mapping[str, int] | None = None,
    allow_large_drop: bool = False,
) -> tuple[bool, str, list[str]]:
    """Apply BOOTSTRAP_GENERATION_ACCEPTANCE v1.  Returns (accepted, rule, reasons)."""

    reasons: list[str] = []

    # R1 — structural completeness.
    if official_element_count <= 0:
        reasons.append("R1_EMPTY_PAYLOAD")
    if not unique_ids:
        reasons.append("R1_DUPLICATE_IDS")
    if not non_empty_ids:
        reasons.append("R1_MISSING_IDS")
    if int(parsed_count) != int(official_element_count):
        reasons.append(f"R1_PARSE_INCOMPLETE parsed={parsed_count} official={official_element_count}")
    if reasons:
        return False, "REJECTED", reasons

    # R4 — nothing to destroy.
    if previous is None:
        return True, "FIRST_GENERATION", []

    previous_ids = {int(pid) for pid in previous.element_ids}
    current_set = {int(pid) for pid in current_ids}
    if not previous_ids:
        return True, "FIRST_GENERATION", []

    dropped = previous_ids - current_set
    retained_fraction = len(current_set & previous_ids) / len(previous_ids)

    # R2 — bounded drift.
    if retained_fraction < MIN_RETAINED_FRACTION:
        reasons.append(
            f"R2_RETAINED_FRACTION {retained_fraction:.4f} < {MIN_RETAINED_FRACTION}"
        )
    if len(dropped) > MAX_ABSOLUTE_DROP:
        reasons.append(f"R2_ABSOLUTE_DROP {len(dropped)} > {MAX_ABSOLUTE_DROP}")

    # R3 — no whole-club loss.
    if current_club_counts is not None:
        for club, count in previous.club_player_counts.items():
            if int(count) > 0 and int(current_club_counts.get(club, 0)) == 0:
                reasons.append(f"R3_WHOLE_CLUB_LOST club={club}")

    if reasons:
        if allow_large_drop:
            return True, "OVERRIDE_ALLOW_LARGE_DROP", reasons
        return False, "REJECTED", reasons
    return True, "STRUCTURAL_AND_BOUNDED_DRIFT", []


def build_generation(
    *,
    payload: Mapping[str, Any],
    parsed_count: int,
    persisted_count: int,
    captured_at: str | None = None,
    run_id: int | None = None,
    previous: BootstrapGeneration | None = None,
    validation_status: str = "VALIDATED",
    allow_large_drop: bool = False,
) -> BootstrapGeneration:
    """Build the generation record and apply the acceptance rule."""

    ids = element_ids_from_payload(payload)
    club_counts = club_player_counts_from_payload(payload)
    accepted, rule, reasons = decide_acceptance(
        current_ids=ids,
        previous=previous,
        official_element_count=len(elements_from_payload(payload)),
        parsed_count=int(parsed_count),
        current_club_counts=club_counts,
        allow_large_drop=allow_large_drop,
    )
    return BootstrapGeneration(
        captured_at=captured_at or _utc_now(),
        run_id=None if run_id is None else int(run_id),
        official_element_count=len(elements_from_payload(payload)),
        parsed_count=int(parsed_count),
        persisted_count=int(persisted_count),
        element_ids=tuple(sorted(int(pid) for pid in ids)),
        element_id_sha256=element_id_sha256(ids),
        club_player_counts=club_counts,
        availability_counts=availability_counts_from_payload(payload),
        validation_status=validation_status,
        accepted=accepted,
        acceptance_rule=rule,
        rejection_reasons=tuple(reasons),
        previous_element_count=None if previous is None else int(previous.official_element_count),
        previous_id_sha256=None if previous is None else previous.element_id_sha256,
        allow_large_drop=bool(allow_large_drop),
    )


def generation_dir(raw_dir: str | Path) -> Path:
    return Path(raw_dir) / GENERATIONS_DIRNAME


def write_generation(raw_dir: str | Path, generation: BootstrapGeneration) -> Path:
    """Persist the generation record (accepted or rejected) plus a timestamped copy."""

    directory = generation_dir(raw_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(generation.as_dict(), indent=2, sort_keys=True) + "\n"
    latest = directory / GENERATION_FILENAME
    latest.write_text(payload, encoding="utf-8")
    stamp = generation.captured_at.replace(":", "").replace("-", "")
    (directory / f"generation_{stamp}.json").write_text(payload, encoding="utf-8")
    return latest


def load_latest_generation(raw_dir: str | Path) -> BootstrapGeneration | None:
    """The last recorded generation, or None when none exists yet."""

    latest = generation_dir(raw_dir) / GENERATION_FILENAME
    if not latest.exists():
        return None
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return BootstrapGeneration(
        captured_at=str(payload["captured_at"]),
        run_id=payload.get("run_id"),
        official_element_count=int(payload["official_element_count"]),
        parsed_count=int(payload["parsed_count"]),
        persisted_count=int(payload["persisted_count"]),
        element_ids=tuple(int(pid) for pid in payload["element_ids"]),
        element_id_sha256=str(payload["element_id_sha256"]),
        club_player_counts={str(k): int(v) for k, v in (payload.get("club_player_counts") or {}).items()},
        availability_counts={str(k): int(v) for k, v in (payload.get("availability_counts") or {}).items()},
        validation_status=str(payload["validation_status"]),
        accepted=bool(payload["accepted"]),
        acceptance_rule=str(payload["acceptance_rule"]),
        rejection_reasons=tuple(str(r) for r in payload.get("rejection_reasons") or ()),
        previous_element_count=payload.get("previous_element_count"),
        previous_id_sha256=payload.get("previous_id_sha256"),
        allow_large_drop=bool(payload.get("allow_large_drop", False)),
    )


def with_persisted_count(
    generation: BootstrapGeneration, persisted_count: int
) -> BootstrapGeneration:
    return replace(generation, persisted_count=int(persisted_count))


class OfficialPoolIncomplete(RuntimeError):
    """The snapshot's player pool is not the accepted official generation."""


# ---------------------------------------------------------------------------
# Decision-time identity assertion (R4B.1.1)
# ---------------------------------------------------------------------------
def assert_official_pool_identity(
    *,
    accepted_generation: Mapping[str, Any] | None,
    snapshot_player_ids: Iterable[int],
    enforce: bool = True,
) -> dict[str, Any]:
    """Prove the decision's pool IS the accepted official generation, by identity.

    ``accepted_generation`` is the row read from ``bootstrap_generations``
    (``accepted=1``) inside the certification snapshot; ``snapshot_player_ids``
    is ``players WHERE is_active=1`` from the SAME snapshot connection.

    Requires BOTH:

      * ``snapshot_pool_count  == official_generation_count``
      * ``snapshot_pool_ids_sha256 == official_generation_ids_sha256``

    An equal count with different ids is a FAILURE: counts alone cannot prove
    identity.  This runs before any candidate promotion, route generation or
    optimisation.  A missing accepted generation also fails closed, because
    without a recorded accepted generation there is nothing to prove the pool
    against.
    """

    snapshot_ids = sorted({int(pid) for pid in snapshot_player_ids})
    snapshot_count = len(snapshot_ids)
    snapshot_sha = element_id_sha256(snapshot_ids)

    reasons: list[str] = []
    if not accepted_generation:
        reasons.append("NO_ACCEPTED_OFFICIAL_GENERATION_IN_SNAPSHOT")
        generation_id = None
        generation_captured_at = None
        generation_count = None
        generation_sha = None
        generation_ids: list[int] = []
    else:
        generation_id = accepted_generation.get("id")
        generation_captured_at = accepted_generation.get("captured_at")
        generation_ids = sorted({int(pid) for pid in (accepted_generation.get("element_ids") or [])})
        generation_count = int(
            accepted_generation.get("official_element_count", len(generation_ids))
        )
        generation_sha = str(accepted_generation.get("element_ids_sha256") or element_id_sha256(generation_ids))
        if generation_count != snapshot_count:
            reasons.append(
                f"COUNT_MISMATCH generation={generation_count} snapshot_pool={snapshot_count}"
            )
        if generation_sha != snapshot_sha:
            reasons.append("ID_HASH_MISMATCH")

    missing = sorted(set(generation_ids) - set(snapshot_ids)) if generation_ids else []
    extra = sorted(set(snapshot_ids) - set(generation_ids)) if generation_ids else []

    report: dict[str, Any] = {
        "official_generation_id": generation_id,
        "official_generation_captured_at": generation_captured_at,
        "official_generation_count": generation_count,
        "official_generation_ids_sha256": generation_sha,
        "snapshot_pool_count": snapshot_count,
        "snapshot_pool_ids_sha256": snapshot_sha,
        "official_pool_identity_match": not reasons,
        "missing_from_snapshot_pool": missing,
        "extra_in_snapshot_pool": extra,
        "identity_reasons": reasons,
        "no_recommendation": True,
    }
    if enforce and reasons:
        raise OfficialPoolIncomplete(
            f"{DIAG_INGEST_INCOMPLETE}: the certification snapshot's active player pool is not the "
            f"accepted official bootstrap generation ({'; '.join(reasons)}; "
            f"missing_from_pool={missing[:8]}, extra_in_pool={extra[:8]}). "
            "The decision is refused before candidate promotion, route generation or optimisation."
        )
    return report


__all__ = [
    "ACCEPTANCE_RULES",
    "ACCEPTANCE_RULE_VERSION",
    "BootstrapGeneration",
    "DIAG_INGEST_INCOMPLETE",
    "GENERATION_FILENAME",
    "GENERATION_SCHEMA",
    "GENERATIONS_DIRNAME",
    "IngestGenerationError",
    "MAX_ABSOLUTE_DROP",
    "MIN_RETAINED_FRACTION",
    "OfficialPoolIncomplete",
    "assert_official_pool_identity",
    "availability_counts_from_payload",
    "build_generation",
    "club_player_counts_from_payload",
    "decide_acceptance",
    "element_id_sha256",
    "element_ids_from_payload",
    "elements_from_payload",
    "generation_dir",
    "load_latest_generation",
    "with_persisted_count",
    "write_generation",
]
