"""Run recording and checkpointing.

Writes three things next to each other:

* ``manifest.json``  provenance
* ``metrics.jsonl``  one machine-readable record per iteration
* ``progress.tsv``   one headed table row per iteration (Excel-friendly)

plus ``state.json`` / ``state.npz`` for resumable training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError("not JSON serializable: %r" % (type(obj),))


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        return "%.6g" % float(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=_json_default).replace("\t", " ").replace("\n", " ")
    return str(value).replace("\t", " ").replace("\n", " ")


class RunRecorder:
    def __init__(self, out_dir: str | Path, resume: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.progress_path = self.out_dir / "progress.tsv"
        self.state_path = self.out_dir / "state.json"
        self.arrays_path = self.out_dir / "state.npz"
        self._header = None
        if resume and self.progress_path.exists() and self.progress_path.stat().st_size > 0:
            first = self.progress_path.read_text(encoding="utf-8").splitlines()[0]
            self._header = first.split("\t")

    # ------------------------------------------------------------- logging
    def log(self, record: Mapping[str, Any]):
        rec = dict(record)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")

        if self._header is None:
            self._header = list(rec.keys())
            with self.progress_path.open("w", encoding="utf-8") as f:
                f.write("\t".join(self._header) + "\n")

        row = [_cell(rec.get(k)) for k in self._header]
        # If a later record carries keys not in the header, append them once.
        extra = [k for k in rec.keys() if k not in self._header]
        if extra:
            self._header.extend(extra)
            lines = self.progress_path.read_text(encoding="utf-8").splitlines()
            lines[0] = "\t".join(self._header)
            self.progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            row.extend(_cell(rec.get(k)) for k in extra)
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write("\t".join(row) + "\n")

    # --------------------------------------------------------- checkpointing
    def save_state(self, state: Mapping[str, Any], arrays: Mapping[str, np.ndarray] | None = None):
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(state), ensure_ascii=False, indent=2,
                                  default=_json_default), encoding="utf-8")
        tmp.replace(self.state_path)
        if arrays is not None:
            tmp_npz = self.arrays_path.with_suffix(".npz.tmp")
            with tmp_npz.open("wb") as f:
                np.savez(f, **{k: np.asarray(v) for k, v in arrays.items()})
            tmp_npz.replace(self.arrays_path)

    def load_state(self):
        if not self.state_path.exists():
            return {}, {}
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        arrays = {}
        if self.arrays_path.exists():
            with np.load(str(self.arrays_path), allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
        return state, arrays

    # ------------------------------------------------------------- snapshots
    def save_npz(self, name: str, arrays: Mapping[str, np.ndarray]):
        path = self.out_dir / (name if name.endswith(".npz") else name + ".npz")
        with path.open("wb") as f:
            np.savez(f, **{k: np.asarray(v) for k, v in arrays.items()})
        return path

    def load_npz(self, name: str):
        path = self.out_dir / (name if name.endswith(".npz") else name + ".npz")
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}
