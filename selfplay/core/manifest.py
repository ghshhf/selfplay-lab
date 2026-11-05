"""Run provenance.

Every run should answer: which code, which seed, which config, which
environment.  This is the minimum needed to make a result re-derivable.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def git_commit(repo_root: str | Path | None = None) -> str | None:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def make_run_id(prefix: str = "run", seed: int | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tail = uuid.uuid4().hex[:6]
    if seed is None:
        return "%s-%s-%s" % (prefix, stamp, tail)
    return "%s-%s-s%d-%s" % (prefix, stamp, int(seed), tail)


@dataclass
class RunManifest:
    run_id: str
    seed: int
    config: dict[str, Any]
    command: str = ""
    repo_root: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = field(default_factory=lambda: None)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    numpy: str = field(default_factory=lambda: np.__version__)
    platform: str = field(default_factory=platform.platform)

    def __post_init__(self):
        if self.git_commit is None:
            self.git_commit = git_commit(self.repo_root)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "seed": int(self.seed),
            "config": self.config,
            "command": self.command,
            "repo_root": self.repo_root,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "python": self.python,
            "numpy": self.numpy,
            "platform": self.platform,
        }

    def write(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
