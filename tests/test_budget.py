"""Tests for the compute ledger (Elo conversions and per-CPU-second skill)."""

import unittest

from selfplay.core.budget import ComputeLedger, elo_from_score, score_from_elo


class EloTest(unittest.TestCase):
    def test_even_score_is_zero(self):
        self.assertAlmostEqual(elo_from_score(0.5), 0.0, places=6)
        self.assertAlmostEqual(score_from_elo(0.0), 0.5, places=6)

    def test_monotonic_and_roundtrip(self):
        scores = [0.2, 0.5, 0.7, 0.9]
        elos = [elo_from_score(s) for s in scores]
        self.assertEqual(elos, sorted(elos))
        for s, e in zip(scores, elos):
            self.assertAlmostEqual(score_from_elo(e), s, places=6)

    def test_perfect_and_zero_scores_stay_finite(self):
        for s in (0.0, 1.0):
            self.assertTrue(abs(elo_from_score(s)) < 5000.0)


class LedgerTest(unittest.TestCase):
    def test_phase_time_and_throughput(self):
        ledger = ComputeLedger()
        ledger.start_iteration()
        with ledger.phase("collect"):
            sum(range(2000))
        with ledger.phase("eval"):
            sum(range(2000))
        out = ledger.finish_iteration(env_frames=1200)
        self.assertIn("collect", out["phase_sec"])
        self.assertGreater(out["frames_per_sec"], 0.0)
        self.assertEqual(out["train_frames"], 1200)
        self.assertGreaterEqual(out["wall_sec"], 0.0)

    def test_first_measurement_defines_zero_then_grows(self):
        ledger = ComputeLedger()
        with ledger.phase("collect"):
            sum(range(3000))
        ledger.finish_iteration(1000)
        first = ledger.skill(0.85)
        self.assertEqual(first["elo_delta"], 0.0)
        with ledger.phase("collect"):
            sum(range(3000))
        ledger.finish_iteration(1000)
        second = ledger.skill(0.92)
        self.assertGreater(second["elo_delta"], 0.0)
        self.assertGreater(second["elo_per_kcs"], 0.0)

    def test_state_roundtrip(self):
        a = ComputeLedger()
        with a.phase("collect"):
            pass
        a.finish_iteration(10)
        b = ComputeLedger()
        b.load_state_dict(a.state_dict())
        self.assertAlmostEqual(a.cpu_s_total, b.cpu_s_total, places=9)
        self.assertEqual(a.frames_total, b.frames_total)


if __name__ == "__main__":
    unittest.main()
