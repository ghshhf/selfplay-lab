"""Paired A/B report between two runs.

Usage::

    python tools/ab.py --a runs/exp_a --b runs/exp_b --metric holdout_win

Both runs must use the same ``--eval-episodes`` and the same holdout level:
``_evaluate`` scores every checkpoint on the fixed suite (seeds 2000.., both
sides), so iteration *i* is a genuine pair, not two independent samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selfplay.core.stats import paired_delta  # noqa: E402


def read_metric(run_dir, metric):
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        raise SystemExit("no metrics.jsonl in %s" % run_dir)
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        value = rec.get(metric)
        if value is None:
            continue
        out[int(rec["iter"])] = float(value)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="treatment run dir")
    ap.add_argument("--b", required=True, help="baseline run dir")
    ap.add_argument("--metric", default="holdout_win")
    ap.add_argument("--from-iter", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=800)
    args = ap.parse_args(argv)

    a = read_metric(args.a, args.metric)
    b = read_metric(args.b, args.metric)
    keys = sorted(set(a) & set(b))
    keys = [k for k in keys if k >= args.from_iter]
    if not keys:
        raise SystemExit("no shared iterations for metric %s" % args.metric)

    pairs = [(k, a[k], b[k]) for k in keys]
    res = paired_delta([p[1] for p in pairs], [p[2] for p in pairs],
                       n_boot=args.n_boot)
    last_n = min(20, len(pairs))
    tail = paired_delta([p[1] for p in pairs[-last_n:]],
                        [p[2] for p in pairs[-last_n:]], n_boot=args.n_boot)

    print("metric      %s" % args.metric)
    print("iterations  %d..%d  (n=%d)" % (keys[0], keys[-1], len(keys)))
    print("mean A      %.4f" % (sum(p[1] for p in pairs) / len(pairs)))
    print("mean B      %.4f" % (sum(p[2] for p in pairs) / len(pairs)))
    print("paired Δ    %+.4f  95%% CI [%+.4f, %+.4f]  -> %s"
          % (res["delta"], res["ci_low"], res["ci_high"], res["sign"]))
    print("last %-2d Δ  %+.4f  95%% CI [%+.4f, %+.4f]  -> %s"
          % (tail["n"], tail["delta"], tail["ci_low"], tail["ci_high"],
             tail["sign"]))
    if res["sign"] == "tie":
        print("interpretation: no separation at 95%; treat as noise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
