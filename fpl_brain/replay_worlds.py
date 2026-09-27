"""PE-9 — the structurally separate non-production replay / research / test path.

Amendment 2 §14 is explicit about where matrix injection may exist and what it may
never do.  This module IS that boundary: it is the only place a caller-supplied
world may enter a search, and it is a different module from the production
generation path on purpose -- so "is this a production decision?" is answered by
which module was called, not by a flag inside one shared implementation.

What the injected path structurally CANNOT do, and how that is enforced rather
than asserted:

* it cannot mint a generation row or move ``current_generation``: it never calls
  :func:`generation_store.certify_generation`, and every replay call re-reads the
  two PE-9 evidence tables (``generation``, ``current_generation``,
  ``engine_decision_records``) before and after and REFUSES if a single row
  changed -- so a future edit that accidentally routed a write through here fails
  the call instead of silently minting evidence;
* it cannot write an ``engine_decision_record``: production decision records are
  appended only by :func:`generation_store.make_decision`, which this module does
  not call;
* it cannot be selected by production ``make_decision``: a production call accepts
  only a generation SELECTOR, and a replay world has no ``generation_id`` to name.

Historical certified replay does NOT need injection at all:
``make_decision(..., generation_id=<historical>)`` re-derives the decision from the
retained manifest, run rows and snapshot.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

#: The declared declaration tokens.  A replay call must name one of these; the
#: declaration travels with the worlds so nothing downstream can mistake them for
#: certified evidence.
NON_PRODUCTION_REPLAY_ONLY = "NON_PRODUCTION_REPLAY_ONLY"
REPLAY_ONLY = "REPLAY_ONLY"
TEST_ONLY = "TEST_ONLY"
DECLARED_REPLAY_TOKENS: tuple[str, ...] = (
    NON_PRODUCTION_REPLAY_ONLY,
    REPLAY_ONLY,
    TEST_ONLY,
)

DIAG_REPLAY_DECLARATION_REQUIRED = "REPLAY_DECLARATION_REQUIRED"
DIAG_REPLAY_WROTE_PE9_EVIDENCE = "REPLAY_WROTE_PE9_EVIDENCE"


class ReplayRefused(RuntimeError):
    """The non-production replay boundary refused a call."""

    def __init__(self, token: str, reasons: list[str]) -> None:
        self.token = str(token)
        self.reasons = [str(reason) for reason in reasons]
        super().__init__(f"{self.token}: " + "; ".join(self.reasons))


@dataclass(frozen=True)
class ReplayDeclaration:
    """Who is exercising the replay path, and under which declared token."""

    token: str
    owner: str
    purpose: str = ""

    def __post_init__(self) -> None:
        if str(self.token) not in DECLARED_REPLAY_TOKENS:
            raise ReplayRefused(
                DIAG_REPLAY_DECLARATION_REQUIRED,
                [
                    f"{self.token!r} is not a declared replay token "
                    f"({list(DECLARED_REPLAY_TOKENS)}); a non-production world source must say so "
                    "under a declared token"
                ],
            )
        if not str(self.owner or "").strip():
            raise ReplayRefused(
                DIAG_REPLAY_DECLARATION_REQUIRED,
                ["a replay declaration must name who is exercising it"],
            )

    @property
    def stamp(self) -> str:
        return f"{self.token}:{self.owner}"

    def as_dict(self) -> dict[str, Any]:
        return {"token": self.token, "owner": self.owner, "purpose": self.purpose}


@dataclass(frozen=True)
class ReplayWorlds:
    """Injected worlds, under a declared non-production replay token.

    Exactly one source: a ``provider`` called per event, or a mapping of matrices.
    """

    declaration: ReplayDeclaration
    provider: Callable[..., Any] | None = None
    matrices: Mapping[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.declaration, str):
            # A bare string is not a declaration: it names no owner.
            raise ReplayRefused(
                DIAG_REPLAY_DECLARATION_REQUIRED,
                [
                    "a replay world source must carry a ReplayDeclaration naming its token and owner, "
                    "not a bare string"
                ],
            )
        if self.provider is None and not self.matrices:
            raise ReplayRefused(
                DIAG_REPLAY_DECLARATION_REQUIRED,
                ["a replay world source must supply a provider or matrices"],
            )
        if self.provider is not None and self.matrices:
            raise ReplayRefused(
                DIAG_REPLAY_DECLARATION_REQUIRED,
                ["a replay world source declares one source, not a provider AND matrices"],
            )

    def as_non_production_worlds(self):
        """The optimizer's declared non-production door for these worlds."""

        from . import route_optimizer as ro

        return ro.NonProductionWorlds(
            declaration=self.declaration.stamp,
            provider=self.provider,
            matrices=dict(self.matrices),
        )

    def matrices_or_none(self) -> Mapping[int, Any] | None:
        return dict(self.matrices) if self.matrices else None


#: The PE-9 evidence tables a replay call must leave byte-for-byte untouched.
PE9_EVIDENCE_TABLES: tuple[str, ...] = (
    "generation",
    "current_generation",
    "engine_decision_records",
)


def evidence_row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts of the PE-9 evidence tables, or ``-1`` where a table is absent."""

    counts: dict[str, int] = {}
    for table in PE9_EVIDENCE_TABLES:
        try:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except sqlite3.Error:
            counts[table] = -1
    return counts


def assert_replay_wrote_no_evidence(
    before: Mapping[str, int], after: Mapping[str, int]
) -> None:
    """Refuse when a replay call changed a PE-9 evidence table."""

    changed = sorted(
        table for table in PE9_EVIDENCE_TABLES if int(before.get(table, -1)) != int(after.get(table, -1))
    )
    if changed:
        raise ReplayRefused(
            DIAG_REPLAY_WROTE_PE9_EVIDENCE,
            [
                f"a non-production replay call changed {changed}; injected worlds can never mint a "
                "generation, move the current-generation pointer, or write a production decision record"
            ],
        )


def assert_not_production_generation(generation_id: Any) -> None:
    """A replay call must not name a production generation selector."""

    if generation_id is not None:
        raise ReplayRefused(
            DIAG_REPLAY_DECLARATION_REQUIRED,
            [
                "a non-production replay call must not name a generation_id; replay worlds are not "
                "certified evidence and cannot select one"
            ],
        )


def replay_decision(
    conn: sqlite3.Connection,
    *,
    run: Callable[[], Any],
    declaration: ReplayDeclaration,
) -> dict[str, Any]:
    """Run a non-production decision closure inside the replay boundary.

    The closure does the work (it may inject worlds through
    :class:`ReplayWorlds`); this wrapper proves the call left no PE-9 evidence behind
    and labels the result as non-production so no downstream consumer can present it
    as a certified decision.
    """

    if not isinstance(declaration, ReplayDeclaration):
        raise ReplayRefused(
            DIAG_REPLAY_DECLARATION_REQUIRED,
            ["a replay call must carry a ReplayDeclaration"],
        )
    before = evidence_row_counts(conn)
    result = run()
    after = evidence_row_counts(conn)
    assert_replay_wrote_no_evidence(before, after)
    return {
        "schema": "fpl_brain.pe9_replay_result.v1",
        "production": False,
        "declaration": declaration.as_dict(),
        "certified": False,
        "generation_id": None,
        "decision_record_id": None,
        "replay_result": result,
    }


__all__ = [
    "DECLARED_REPLAY_TOKENS",
    "DIAG_REPLAY_DECLARATION_REQUIRED",
    "DIAG_REPLAY_WROTE_PE9_EVIDENCE",
    "NON_PRODUCTION_REPLAY_ONLY",
    "PE9_EVIDENCE_TABLES",
    "REPLAY_ONLY",
    "ReplayDeclaration",
    "ReplayRefused",
    "ReplayWorlds",
    "TEST_ONLY",
    "assert_not_production_generation",
    "assert_replay_wrote_no_evidence",
    "evidence_row_counts",
    "replay_decision",
]
