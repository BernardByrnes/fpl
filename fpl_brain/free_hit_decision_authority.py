"""Genuine Free Hit decision authority — derived from the REAL certification artifact.

WHY THIS MODULE EXISTS
----------------------
Agreement between caller-owned objects is not certification.  If the Free Hit
request is allowed to carry its own ``FreeHitDecisionAuthority``, a caller can
create an authority B, a world identity B and a binding identity B that all agree
with each other, and move the cutoff, the source identity, the model versions and
the certified horizon together.  Nothing about that is anchored to anything.

The authority is therefore DERIVED from the canonical certification artifact the
accepted four-GW pipeline already produces, loaded through the accepted loader
(``four_gw_decision.load_certification_artifact``) and verified by RECOMPUTING
``four_gw_decision.certification_identity_of`` from the artifact's own cutoff,
bundle identities and data snapshot.  A copied identity on a different artifact
fails; mutating one certified bundle fails.

THE REAL SCHEMA, USED AS-IS
---------------------------
The artifact carries ``planning_cutoff``, ``events``, ``data_snapshot_sha256``,
``code_snapshot_sha256``, ``model_versions``, ``certified_bundles`` (one per
event) and ``certified_bundle_identity``.  Each certified bundle commits to
``cutoff``, ``data_snapshot_sha256``, ``code_snapshot_sha256``, ``runs``,
``model_versions`` and ``planning_context_hash``.

No Free-Hit-only field is invented.  Predictive authority is mapped from what the
certifier actually commits to, through canonical labels defined ONCE here
(``runs_label`` / ``model_label``) so a producer of a world identity and this
verifier cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

FH_DECISION_AUTHORITY_REQUIRED = "FREE_HIT_CANONICAL_DECISION_AUTHORITY_REQUIRED"
FH_DECISION_AUTHORITY_MISMATCH = "FREE_HIT_PREDICTIVE_EVIDENCE_NOT_THE_CERTIFIED_ONE"
FH_CERTIFIED_EVENTS_MISMATCH = "FREE_HIT_CERTIFIED_EVENT_SET_MISMATCH"
FH_CERTIFIED_CUTOFF_MISMATCH = "FREE_HIT_CERTIFIED_CUTOFF_MISMATCH"
FH_CERTIFIED_SNAPSHOT_MISMATCH = "FREE_HIT_CERTIFIED_DATA_SNAPSHOT_MISMATCH"


class FreeHitAuthorityError(ValueError):
    """The canonical certification could not be loaded or does not apply."""

    def __init__(self, message: str, *, reasons: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.reasons = tuple(str(reason) for reason in reasons)


def _digest(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def runs_label(runs: Mapping[str, Any] | None) -> str:
    """The canonical label of a bundle's projection RUNS.

    ONE definition, shared by whatever builds a world identity and by this
    verifier, so the two cannot drift on how a set of run ids is named.
    """

    return _digest({str(k): int(v) for k, v in (runs or {}).items()})


def model_label(model_versions: Any) -> str:
    """The canonical label of a bundle's MODEL VERSIONS."""

    if isinstance(model_versions, Mapping):
        canonical = {str(k): str(v) for k, v in model_versions.items()}
    else:
        canonical = {str(row.get("model_family")): str(row.get("model_version"))
                     for row in (model_versions or ()) if isinstance(row, Mapping)}
    return _digest(canonical)


@dataclass(frozen=True)
class FreeHitDecisionAuthority:
    """The CANONICAL certified decision context, derived from a loaded artifact.

    Every field comes from the artifact; none is a caller assertion.  It is built
    only by :meth:`from_certification`, which verifies the artifact through the
    accepted loader and recomputes the certification identity.
    """

    planning_cutoff: str
    data_snapshot_sha256: str
    certification_identity: str
    certified_events: tuple[int, ...]
    #: Per-event DECLARED bundle identity (what the certification identity hashes).
    certified_bundle_identity: Mapping[int, str] = field(default_factory=dict)
    code_snapshot_sha256: str = ""
    model_versions: tuple[tuple[str, str], ...] = ()
    planning_context_hash: str = ""
    #: Where the authority was loaded from.  Empty means it was NOT loaded from
    #: canonical certification, which is exactly what production refuses.
    loaded_from: str = ""
    #: The certified bundles themselves, keyed by event.  Predictive authority is
    #: read from THESE, because they are what the certifier actually commits to.
    bundle_map: Mapping[int, Mapping[str, Any]] = field(default_factory=dict, compare=False)
    #: The LOADED artifact the authority was derived from.  It travels with the
    #: authority because a downstream predictive load must present the
    #: AUTHORISATION, not merely run ids that happen to agree with it: the world
    #: loader compares the bundle it is handed against the artifact's own recorded
    #: bundle.  Never compared, never serialized, never part of the identity.
    artifact: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def problems(self) -> list[str]:
        found: list[str] = []
        for name in ("planning_cutoff", "data_snapshot_sha256", "certification_identity",
                     "code_snapshot_sha256", "planning_context_hash"):
            if not str(getattr(self, name) or "").strip():
                found.append(f"the certified decision authority carries no {name}")
        if not self.certified_events:
            found.append("the certified decision authority certifies no events")
        if not self.certified_bundle_identity:
            found.append("the certified decision authority carries no bundle identities")
        if not str(self.loaded_from or "").strip():
            found.append("the authority was not derived from a loaded certification artifact")
        return found

    def as_dict(self) -> dict[str, Any]:
        return {
            "planning_cutoff": str(self.planning_cutoff),
            "data_snapshot_sha256": str(self.data_snapshot_sha256),
            "certification_identity": str(self.certification_identity),
            "certified_events": list(self.certified_events),
            "certified_bundle_identity": {str(k): str(v) for k, v in sorted(self.certified_bundle_identity.items())},
            "code_snapshot_sha256": str(self.code_snapshot_sha256),
            "model_versions": [list(row) for row in self.model_versions],
            "planning_context_hash": str(self.planning_context_hash),
            "loaded_from": str(self.loaded_from),
        }

    def bundle_for(self, event: int) -> Mapping[str, Any] | None:
        return self.bundle_map.get(int(event))

    def disagreements_with(
        self, identity: Any, *, event: int | None = None, label: str = "identity"
    ) -> list[str]:
        """Every predictive dimension on which ``identity`` is not the certified one.

        The certified values come from the CERTIFIED BUNDLE for the event, which is
        what the certifier actually commits to -- not from invented Free-Hit fields.
        """

        found: list[str] = []
        if identity is None:
            return [f"{label} carries no predictive identity"]
        bundle = self.bundle_for(int(event)) if event is not None else None
        cutoff = str((bundle or {}).get("cutoff") or self.planning_cutoff)
        snapshot = str((bundle or {}).get("data_snapshot_sha256") or self.data_snapshot_sha256)
        code = str((bundle or {}).get("code_snapshot_sha256") or self.code_snapshot_sha256)
        runs = (bundle or {}).get("runs")
        models = (bundle or {}).get("model_versions")
        pairs = (
            ("cutoff", str(getattr(identity, "cutoff", "")), cutoff),
            ("data_snapshot_sha256", str(getattr(identity, "data_snapshot_sha256", "")), snapshot),
            ("source_snapshot_sha256", str(getattr(identity, "source_snapshot_sha256", "")), code),
            ("generation", str(getattr(identity, "generation", "")),
             runs_label(runs) if runs is not None else ""),
            ("model_config_identity", str(getattr(identity, "model_config_identity", "")),
             model_label(models) if models is not None else ""),
        )
        for name, supplied, certified in pairs:
            if certified and supplied != certified:
                found.append(f"{label} {name}")
        return found

    def certified_events_problems(self, events: Sequence[int]) -> list[str]:
        """The FH horizon must BE the certified event set, exactly."""

        expected = tuple(int(e) for e in events)
        certified = tuple(int(e) for e in self.certified_events)
        if expected != certified:
            return [
                f"{FH_CERTIFIED_EVENTS_MISMATCH}: requested events {list(expected)} are not the "
                f"certified events {list(certified)}"
            ]
        return []

    @classmethod
    def from_certification(
        cls, artifact: Mapping[str, Any], *, loaded_from: str = ""
    ) -> "FreeHitDecisionAuthority":
        """Derive the authority from a REAL certification artifact, or refuse.

        The artifact's ``four_gw_certification_identity`` must equal the value
        RECOMPUTED from its own cutoff, bundle identities and data snapshot, so a
        recognised identity copied onto a different artifact fails and mutating one
        certified bundle fails.
        """

        from . import four_gw_decision as fg

        if not artifact:
            raise FreeHitAuthorityError(
                f"{FH_DECISION_AUTHORITY_REQUIRED}: no certification artifact supplied",
                reasons=(FH_DECISION_AUTHORITY_REQUIRED,),
            )
        declared = str(
            artifact.get("four_gw_certification_identity") or artifact.get("certification_identity") or ""
        )
        if not declared:
            raise FreeHitAuthorityError(
                f"{FH_DECISION_AUTHORITY_REQUIRED}: the certification artifact declares no identity",
                reasons=(FH_DECISION_AUTHORITY_REQUIRED,),
            )
        recomputed = str(fg.certification_identity_of(artifact))
        if declared != recomputed:
            raise FreeHitAuthorityError(
                f"{FH_DECISION_AUTHORITY_REQUIRED}: the certification artifact's declared identity is not "
                "the one its own cutoff, bundle identities and data snapshot produce",
                reasons=(FH_DECISION_AUTHORITY_REQUIRED,),
            )

        bundles = artifact.get("certified_bundles") or {}
        declared_bundles = artifact.get("certified_bundle_identity") or {}
        bundle_map = {int(event): dict(row) for event, row in bundles.items() if isinstance(row, Mapping)}
        first = bundle_map[min(bundle_map)] if bundle_map else {}
        models = artifact.get("model_versions") or first.get("model_versions") or {}
        if isinstance(models, Sequence) and not isinstance(models, (str, bytes, Mapping)):
            model_rows = tuple(
                (str(row.get("model_family")), str(row.get("model_version")))
                for row in models if isinstance(row, Mapping)
            )
        else:
            model_rows = tuple((str(k), str(v)) for k, v in dict(models).items())

        authority = cls(
            planning_cutoff=str(artifact.get("planning_cutoff") or ""),
            data_snapshot_sha256=str(artifact.get("data_snapshot_sha256") or ""),
            certification_identity=str(declared),
            certified_events=tuple(int(event) for event in (artifact.get("events") or ())),
            certified_bundle_identity={
                int(event): str(identity) for event, identity in dict(declared_bundles).items()
            },
            code_snapshot_sha256=str(artifact.get("code_snapshot_sha256") or first.get("code_snapshot_sha256") or ""),
            model_versions=model_rows,
            planning_context_hash=str(first.get("planning_context_hash") or ""),
            loaded_from=str(loaded_from or "in-memory artifact"),
            bundle_map=bundle_map,
            artifact=artifact,
        )
        problems = authority.problems()
        if problems:
            raise FreeHitAuthorityError(
                f"{FH_DECISION_AUTHORITY_REQUIRED}: " + "; ".join(problems[:6]),
                reasons=(FH_DECISION_AUTHORITY_REQUIRED,),
            )
        return authority


def load_decision_authority(path: Any) -> FreeHitDecisionAuthority:
    """Load the CANONICAL certification artifact through the accepted loader.

    ``four_gw_decision.load_certification_artifact`` is the production loader: it
    enforces the artifact schema, the causal/temporal status, dependency
    coherence, the authorisation flag, the history-completeness audit and the
    certification-wiring identity.  Nothing here re-implements or bypasses it.
    """

    from . import four_gw_decision as fg

    try:
        artifact = fg.load_certification_artifact(path)
    except Exception as exc:  # the canonical loader's own refusals are authoritative
        raise FreeHitAuthorityError(
            f"{FH_DECISION_AUTHORITY_REQUIRED}: the canonical certification loader refused "
            f"{path!r}: {exc}",
            reasons=(FH_DECISION_AUTHORITY_REQUIRED,),
        ) from exc
    return FreeHitDecisionAuthority.from_certification(artifact, loaded_from=str(path))
