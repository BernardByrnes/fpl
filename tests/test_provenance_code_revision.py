"""Code-revision provenance: it must identify the EXECUTING source, not the launcher.

``code_revision`` is written into every projection run, so it is the metadata a
reviewer uses to attribute a prediction to a commit.  Deriving it from the
process working directory silently attributes a run to whatever repository the
operator happened to be standing in, which is exactly what happened when the
PE-2 forward anchor was generated from a second worktree.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from fpl_brain import analytics

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _git(args, cwd, config=(), **kwargs):
    """Run git with the repository-safe config placed before the subcommand."""

    pairs: list[str] = ["-c", "safe.directory=*"]
    for key, value in config:
        pairs += ["-c", f"{key}={value}"]
    return subprocess.run(["git", *pairs, *args], cwd=str(cwd),
                          capture_output=True, text=True, **kwargs)


def source_root() -> Path:
    return Path(analytics.__file__).resolve().parents[1]


def test_code_revision_identifies_the_executing_source():
    """The normal case: the revision is the source tree's HEAD and a real SHA."""

    revision = analytics.code_revision()
    assert revision is not None, "the executing source is inside a git worktree"
    assert SHA_RE.match(revision), revision
    assert revision == _git(["rev-parse", "HEAD"], source_root()).stdout.strip()


def test_code_revision_ignores_the_working_directory(tmp_path, monkeypatch):
    """The regression for the anchor mis-attribution.

    A DIFFERENT git repository is the working directory, with its own HEAD.  The
    recorded revision must still name the repository that contains the executing
    source.  On the previous implementation this returns the decoy's HEAD, and
    the first assertion fails for exactly that reason.
    """

    decoy = tmp_path / "decoy_repo"
    decoy.mkdir()
    assert _git(["init", "-q"], decoy).returncode == 0
    (decoy / "README.md").write_text("decoy\n", encoding="utf-8")
    _git(["add", "."], decoy)
    committed = _git(["commit", "-q", "-m", "decoy"], decoy,
                     config=(("user.email", "decoy@example.invalid"), ("user.name", "decoy")))
    assert committed.returncode == 0, committed.stderr
    decoy_head = _git(["rev-parse", "HEAD"], decoy).stdout.strip()
    assert SHA_RE.match(decoy_head), decoy_head

    expected = _git(["rev-parse", "HEAD"], source_root()).stdout.strip()
    assert decoy_head != expected, "the decoy must have a different HEAD to be a test"

    monkeypatch.chdir(decoy)
    revision = analytics.code_revision()

    assert revision != decoy_head, (
        "code_revision recorded the working directory's repository instead of the "
        "repository containing the executing source"
    )
    assert revision == expected


def test_code_revision_is_stable_regardless_of_the_calling_directory(tmp_path, monkeypatch):
    """Same source, two different working directories, one answer."""

    expected = analytics.code_revision()
    for directory in (source_root(), tmp_path):
        monkeypatch.chdir(directory)
        assert analytics.code_revision() == expected


def test_code_revision_fails_closed_outside_a_git_worktree(monkeypatch):
    """No git, no revision -- never a misleading invented one."""

    def explode(*_args, **_kwargs):
        raise OSError("git is not available")

    monkeypatch.setattr("subprocess.run", explode)
    assert analytics.code_revision() is None


def test_code_revision_returns_none_when_git_reports_no_revision(monkeypatch):
    class _Result:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Result())
    assert analytics.code_revision() is None


def test_code_revision_is_distinct_from_the_source_snapshot_identity():
    """The two identities answer different questions and must not be conflated.

    ``code_revision`` is a Git commit; ``source_snapshot_sha256`` is a content
    hash over SOURCE_SNAPSHOT_FILES.  A dirty worktree has a real content
    identity while its commit alone would not describe it.
    """

    revision = analytics.code_revision()
    assert revision is not None and SHA_RE.match(revision)
    assert not revision.startswith("sha256:")
    assert "fpl_brain/analytics.py" in list(analytics.SOURCE_SNAPSHOT_FILES)
    snapshot = analytics.canonical_hash({name: name for name in analytics.SOURCE_SNAPSHOT_FILES})
    assert snapshot.startswith("sha256:")
