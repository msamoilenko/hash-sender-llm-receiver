"""Third-party C sources at pinned commits, checked out under the git-ignored `vendor/`.

The lock (`vendor.lock`, versioned) has one line per repository: name, URL, full commit, license.
Licenses were read from each clone on 2026-11-05.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = ROOT / "vendor"
LOCK = HERE / "vendor.lock"


@dataclass(frozen=True)
class Entry:
    name: str
    url: str
    commit: str
    license: str


def entries() -> list[Entry]:
    return [Entry(*ln.split(maxsplit=3)) for ln in LOCK.read_text().splitlines() if ln.strip()]


def _git(*args: str) -> str:
    res = subprocess.run(["git", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {res.stderr.strip()}")
    return res.stdout.strip()


def ensure() -> Path:
    """Clone missing repositories and check out the pinned commits; fail if HEAD differs."""
    for e in entries():
        path = VENDOR / e.name
        if not path.exists():
            _git("clone", "-q", e.url, str(path))
        if _git("-C", str(path), "rev-parse", "HEAD") != e.commit:
            _git("-C", str(path), "checkout", "-q", e.commit)
        head = _git("-C", str(path), "rev-parse", "HEAD")
        if head != e.commit:
            raise RuntimeError(f"{e.name}: HEAD {head} != pinned {e.commit}")
    return VENDOR
