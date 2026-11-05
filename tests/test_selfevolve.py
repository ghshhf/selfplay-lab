"""Tests for the zero-data self-evolution pool."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NEURO = ROOT / "domains" / "neuro"
if str(NEURO) not in sys.path:
    sys.path.insert(0, str(NEURO))

from selfevolve import (GENE_BOUNDS, GENE_WEAPONS, SelfPlayPool,  # noqa: E402
                        mutate_gene, random_gene)


def _pool():
    p = SelfPlayPool(beta=2.0, alpha=0.25, max_size=6, min_games=2)
    for name in ("random", "a", "b", "c"):
        p.add(name, lambda: object(), origin="random" if name == "random"
              else "fixed")
    return p


class TestRecord(unittest.TestCase):
    def test_running_rate_moves_towards_outcomes(self):
        p = _pool()
        for _ in range(12):
            p.record("a", 1.0)
        self.assertGreater(p.get("a").win, 0.95)
        for _ in range(40):
            p.record("a", 0.0)
        self.assertLess(p.get("a").win, 0.05)
        self.assertEqual(p.get("a").games, 52)

    def test_draw_counts_as_half(self):
        p = _pool()
        p.record("a", 0.5)
        self.assertAlmostEqual(p.get("a").win, 0.5)

    def test_unknown_name_is_ignored(self):
        p = _pool()
        p.record("nobody", 1.0)   # must not raise
        self.assertEqual(len(p), 4)


class TestWeights(unittest.TestCase):
    def test_pfsp_favours_the_opponent_we_lose_to(self):
        p = _pool()
        p.get("a").win = 0.15
        p.get("b").win = 0.85
        p.get("c").win = 0.50
        p.get("random").win = 0.50
        for e in p.entries:
            e.games = 5
        w = p.weights("pfsp")
        self.assertGreater(w[1], w[2])       # a (hard) > b (easy)
        self.assertGreater(w[1], w[3])

    def test_progress_favours_the_moving_frontier(self):
        p = _pool()
        for e in p.entries:
            e.games = 5
        p.get("a").progress = 0.0
        p.get("b").progress = 0.4
        p.get("c").progress = 0.0
        p.get("random").progress = 0.0
        w = p.weights("progress")
        self.assertEqual(int(np.argmax(w)), 2)  # "b"

    def test_cold_start_bonus(self):
        p = _pool()
        for e in p.entries:
            e.games = 5
        p.get("a").win = 0.5
        p.add("fresh", lambda: object(), win=0.5, origin="evolved")
        w = p.weights("uniform")
        self.assertGreater(w[-1], w[1])

    def test_uniform_is_uniform(self):
        p = _pool()
        for e in p.entries:
            e.games = 5
        w = p.weights("uniform")
        self.assertTrue(np.allclose(w, w[0]))

    def test_sampling_prefers_hard_opponent(self):
        p = _pool()
        for e in p.entries:
            e.games = 5
        p.get("a").win = 0.05
        p.get("b").win = 0.95
        p.get("c").win = 0.95
        p.get("random").win = 0.95
        rng = np.random.default_rng(0)
        hits = sum(1 for _ in range(400) if p.sample(rng, "pfsp").name == "a")
        self.assertGreater(hits, 250)


class TestFrontierAndPrune(unittest.TestCase):
    def test_frontier_is_closest_to_half(self):
        p = _pool()
        for e in p.entries:
            e.games = 4
        p.get("a").win = 0.1
        p.get("b").win = 0.55
        p.get("c").win = 0.9
        p.get("random").win = 0.9
        self.assertEqual(p.frontier().name, "b")

    def test_frontier_needs_min_games(self):
        p = _pool()
        p.get("a").win = 0.5
        self.assertIsNone(p.frontier())

    def test_prune_retires_hopeless_but_keeps_random_and_frontier(self):
        p = _pool()
        for e in p.entries:
            e.games = 9
        p.get("a").win = 0.0
        p.get("b").win = 0.5
        p.get("c").win = 0.99
        p.prune()
        self.assertIsNone(p.get("a"))
        self.assertIsNotNone(p.get("b"))
        self.assertIsNotNone(p.get("random"))

    def test_cap_never_exceeds_max_size(self):
        p = _pool()
        for i in range(8):
            p.add("x%d" % i, lambda: object(), origin="evolved")
        p.prune()
        self.assertLessEqual(len(p), 6)

    def test_stats_shape(self):
        p = _pool()
        for e in p.entries:
            e.games = 3
        s = p.stats()
        for key in ("pool_size", "pool_frontier", "pool_frontier_win",
                    "pool_hardest", "pool_games"):
            self.assertIn(key, s)
        self.assertEqual(s["pool_size"], 4)


class TestGenes(unittest.TestCase):
    def test_random_gene_is_inside_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            g = random_gene(rng)
            for k, (lo, hi) in GENE_BOUNDS.items():
                self.assertGreaterEqual(g[k], lo)
                self.assertLessEqual(g[k], hi)
            self.assertIn(g["weapon"], GENE_WEAPONS)
            self.assertIsInstance(g["react"], int)

    def test_mutation_stays_inside_bounds(self):
        rng = np.random.default_rng(1)
        g = random_gene(rng)
        for _ in range(40):
            g = mutate_gene(g, rng, sigma=1.2)
            for k, (lo, hi) in GENE_BOUNDS.items():
                self.assertGreaterEqual(g[k], lo)
                self.assertLessEqual(g[k], hi)
            self.assertIn(g["weapon"], GENE_WEAPONS)

    def test_mutation_preserves_weapon_mostly(self):
        rng = np.random.default_rng(2)
        g = {"weapon": "shield", "aggr": 0.5, "guard": 0.5, "kick": 0.2,
             "dist": 1.2, "react": 8, "idle": 0.2, "loot": 0.2, "avoid": 0.2}
        rng2 = np.random.default_rng(3)
        out = mutate_gene(g, rng2, sigma=0.0, weapon_p=0.0)
        self.assertEqual(out["weapon"], "shield")
        self.assertAlmostEqual(out["aggr"], 0.5)


class TestState(unittest.TestCase):
    def test_stats_round_trip(self):
        p = _pool()
        for _ in range(4):
            p.record("a", 1.0)
        saved = p.state_dict()
        q = _pool()
        q.load_stats(saved)
        self.assertAlmostEqual(q.get("a").win, p.get("a").win)
        self.assertEqual(q.get("a").games, p.get("a").games)

    def test_load_ignores_unknown_names(self):
        p = _pool()
        p.load_stats({"entries": [{"name": "ghost", "win": 0.9, "games": 5}]})
        self.assertIsNone(p.get("ghost"))


class TestTrainerIntegration(unittest.TestCase):
    def test_zero_data_run_grows_its_own_pool(self):
        import importlib
        import shutil
        import tempfile

        import train as T
        tmp = Path(tempfile.mkdtemp())
        try:
            args = T.build_argparser().parse_args([
                "--env", "freeform", "--selfevolve", "--iterations", "3",
                "--episodes", "3", "--max-frames", "220", "--seed", "3",
                "--warmup", "1", "--eval-episodes", "2", "--eval-every", "3",
                "--out-dir", str(tmp),
            ])
            trainer = T.NeuroTrainer(args)
            self.assertIsNone(trainer.bc_loss)      # no demos at all
            self.assertGreaterEqual(len(trainer.pool), 2)
            rec = trainer.train_iteration(0)
            self.assertIn("pool_frontier", rec)
            self.assertGreater(rec["pool_size"], 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
