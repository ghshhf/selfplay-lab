"""Tests for evaluation statistics, weight-space evolution, latent rollout."""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NEURO = ROOT / "domains" / "neuro"
TOOLS = ROOT / "tools"
for p in (ROOT, NEURO, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from selfplay.core.stats import (bootstrap_ci, paired_delta,  # noqa: E402
                                 required_games, summarise, wilson_interval)
from selfplay.nn.latent import LatentWorldModel  # noqa: E402
from selfplay.nn.pbt import PBTPopulation, clone_agent, perturb_params  # noqa: E402
from selfplay.nn.ppo import PPOAgent, PPOConfig  # noqa: E402
from selfplay.nn.rnd import RND  # noqa: E402


def _agent(obs_dim=6, n_actions=3, hidden=8, seed=0):
    return PPOAgent(obs_dim, n_actions, hidden=hidden, cfg=PPOConfig(), seed=seed)


class TestStats(unittest.TestCase):
    def test_bootstrap_brackets_the_mean(self):
        values = [0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
        lo, hi = bootstrap_ci(values, n_boot=300, seed=0)
        self.assertLessEqual(lo, np.mean(values))
        self.assertGreaterEqual(hi, np.mean(values))
        self.assertLess(lo, hi)

    def test_all_same_has_zero_width(self):
        lo, hi = bootstrap_ci([0.5] * 20)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)

    def test_empty_is_safe(self):
        self.assertEqual(bootstrap_ci([]), (0.0, 0.0))
        self.assertEqual(summarise([])["n"], 0)

    def test_paired_delta_detects_consistent_gain(self):
        a = [0.6] * 12
        b = [0.4] * 12
        res = paired_delta(a, b)
        self.assertAlmostEqual(res["delta"], 0.2)
        self.assertEqual(res["sign"], "a")
        self.assertGreater(res["ci_low"], 0.0)

    def test_paired_delta_calls_noise_a_tie(self):
        rng = np.random.default_rng(0)
        a = rng.random(10)
        b = a + rng.normal(0, 0.3, 10)
        res = paired_delta(a, b)
        self.assertEqual(res["sign"], "tie")

    def test_wilson_stays_inside_unit_interval(self):
        for k, n in [(0, 5), (5, 5), (3, 5), (0, 0)]:
            lo, hi = wilson_interval(k, n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, hi)

    def test_required_games(self):
        self.assertGreaterEqual(required_games(0.10), 96)
        self.assertGreater(required_games(0.05), required_games(0.10))


class TestPBT(unittest.TestCase):
    def test_clone_is_identical_without_noise(self):
        a = _agent(seed=1)
        b = clone_agent(a)
        for la, lb in zip(a.policy.params, b.policy.params):
            for pa, pb in zip(la, lb):
                self.assertTrue(np.allclose(pa, pb))
        self.assertIsNot(a, b)

    def test_perturbation_changes_params_but_keeps_shape(self):
        a = _agent(seed=1)
        b = clone_agent(a, sigma=0.05, rng=np.random.default_rng(2))
        changed = 0
        for la, lb in zip(a.policy.params, b.policy.params):
            for pa, pb in zip(la, lb):
                self.assertEqual(pa.shape, pb.shape)
                if not np.allclose(pa, pb):
                    changed += 1
        self.assertGreater(changed, 0)

    def test_relative_noise_scales_with_tensor(self):
        small = [np.full((4, 4), 0.1) * np.arange(16).reshape(4, 4) / 15.0]
        big = [small[0] * 100.0]
        cp_s, cp_b = [p.copy() for p in small], [p.copy() for p in big]
        rng = np.random.default_rng(4)
        perturb_params([small], 0.1, rng)
        perturb_params([big], 0.1, rng)
        self.assertLess(np.abs(small[0] - cp_s[0]).mean(),
                        np.abs(big[0] - cp_b[0]).mean())

    def test_evolve_keeps_the_fittest(self):
        base = _agent(seed=3)
        pop = PBTPopulation(base, size=4, sigma=0.02, seed=5)
        # Fitness keyed to an identifiable marker in the first weight tensor.
        fit = lambda m: float(np.sum(m.policy.params[0][0]))
        best_before = max(pop.members, key=fit)
        champion, report = pop.evolve(fit)
        self.assertGreaterEqual(report["pbt_best"], report["pbt_worst"])
        self.assertEqual(len(pop.members), 4)
        self.assertGreaterEqual(fit(champion), fit(best_before) - 1e-9)

    def test_population_size_is_respected(self):
        pop = PBTPopulation(_agent(seed=0), size=5)
        self.assertEqual(len(pop.members), 5)
        pop.evolve(lambda m: 0.5)
        self.assertEqual(len(pop.members), 5)

    def test_sigma_zero_children_match_parents(self):
        pop = PBTPopulation(_agent(seed=1), size=4, sigma=0.0)
        champ, _ = pop.evolve(lambda m: 0.5)
        for m in pop.members:
            for la, lb in zip(champ.policy.params, m.policy.params):
                for pa, pb in zip(la, lb):
                    self.assertTrue(np.allclose(pa, pb))


class TestLatentRollout(unittest.TestCase):
    def _model(self, latent=4):
        return LatentWorldModel(23, 6, hidden=16, latent=latent,
                                lr=1e-3, seed=0)

    def test_step_latent_shape_and_reward(self):
        m = self._model()
        z, r = m.step_latent(np.zeros(4), 2)
        self.assertEqual(z.shape, (4,))
        self.assertIsInstance(r, float)

    def test_reward_head_learns(self):
        m = self._model()
        rng = np.random.default_rng(0)
        obs = rng.normal(0, 1, (64, 23)).astype(np.float32)
        nxt = rng.normal(0, 1, (64, 23)).astype(np.float32)
        acts = rng.integers(0, 6, 64)
        # reward = action index -> learnable mapping
        rewards = acts.astype(np.float64)
        first = m.train(obs, acts, nxt, rng, epochs=2, rewards=rewards)
        for _ in range(12):
            last = m.train(obs, acts, nxt, rng, epochs=2, rewards=rewards)
        self.assertLess(last["reward_mse"], first["reward_mse"])

    def test_frozen_encoder_does_not_move(self):
        m = self._model()
        rng = np.random.default_rng(1)
        obs = rng.normal(0, 1, (32, 23)).astype(np.float32)
        nxt = rng.normal(0, 1, (32, 23)).astype(np.float32)
        acts = rng.integers(0, 6, 32)
        before = [p.copy() for layer in m.encoder.params for p in layer]
        m.freeze_encoder = True
        m.train(obs, acts, nxt, rng, epochs=3, rewards=np.zeros(32))
        after = [p for layer in m.encoder.params for p in layer]
        for a, b in zip(before, after):
            self.assertTrue(np.allclose(a, b))

    def test_decode_returns_observation_size(self):
        m = self._model()
        out = m.decode(np.zeros(4))
        self.assertEqual(out.shape, (23,))


class TestRND(unittest.TestCase):
    def test_novelty_decays_for_a_repeated_state(self):
        rnd = RND(6, hidden=8, out_dim=4, lr=1e-2, seed=0)
        obs = np.linspace(-1, 1, 6).astype(np.float32)
        first = rnd.error(obs)
        for _ in range(60):
            rnd.update(obs)
        self.assertLess(rnd.error(obs), first * 0.5)

    def test_target_network_is_frozen(self):
        rnd = RND(6, hidden=8, out_dim=4, lr=1e-2, seed=0)
        obs = np.ones(6, dtype=np.float32)
        before = [p.copy() for layer in rnd.target.params for p in layer]
        for _ in range(10):
            rnd.update(obs)
        for a, b in zip(before, [p for layer in rnd.target.params for p in layer]):
            self.assertTrue(np.allclose(a, b))

    def test_state_dict_roundtrip(self):
        rnd = RND(6, hidden=8, out_dim=4, lr=1e-2, seed=0)
        obs = np.linspace(-1, 1, 6).astype(np.float32)
        for _ in range(20):
            rnd.update(obs)
        score = rnd.error(obs)
        clone = RND(6, hidden=8, out_dim=4, lr=1e-2, seed=1)
        clone.load_state_dict(rnd.state_dict(), "rnd")
        self.assertAlmostEqual(clone.error(obs), score, places=5)


class TestTrainerIntegration(unittest.TestCase):
    def _run(self, extra):
        import train as T
        tmp = Path(tempfile.mkdtemp())
        try:
            args = T.build_argparser().parse_args([
                "--env", "freeform", "--iterations", "2", "--episodes", "2",
                "--max-frames", "160", "--eval-episodes", "2", "--eval-every", "1",
                "--warmup", "1", "--out-dir", str(tmp)] + extra)
            trainer = T.NeuroTrainer(args)
            rec = trainer.train_iteration(0)
            return rec, trainer
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ci_fields_present(self):
        rec, _ = self._run([])
        self.assertLessEqual(rec["holdout_ci_low"], rec["holdout_win"])
        self.assertGreaterEqual(rec["holdout_ci_high"], rec["holdout_win"])
        self.assertEqual(len(rec["holdout_outcomes"]), rec["eval_games"])

    def test_latent_policy_shrinks_the_policy_input(self):
        rec, trainer = self._run(["--latent-policy", "--latent-dim", "8"])
        self.assertEqual(rec["policy_dim"], 8)
        self.assertIsNotNone(trainer.encoder)
        self.assertTrue(trainer.world_model.freeze_encoder)

    def test_imagination_adds_steps(self):
        rec, _ = self._run(["--latent-world", "--imagine-steps", "2",
                            "--imagine-batch", "8"])
        self.assertGreater(rec["imagine_steps"], 0)

    def test_pbt_produces_a_scored_population(self):
        rec, trainer = self._run(["--pbt", "--pbt-every", "1", "--pbt-size", "3",
                                  "--pbt-games", "2"])
        self.assertIn("pbt_best", rec)
        self.assertEqual(len(trainer.pbt.members), 3)

    def test_continuous_margin_is_recorded(self):
        rec, _ = self._run([])
        self.assertGreaterEqual(rec["holdout_margin"], -1.0)
        self.assertLessEqual(rec["holdout_margin"], 1.0)
        self.assertLessEqual(rec["holdout_margin_ci_low"], rec["holdout_margin"])
        self.assertGreaterEqual(rec["holdout_margin_ci_high"], rec["holdout_margin"])

    def test_rnd_curiosity_runs(self):
        rec, trainer = self._run(["--latent-policy", "--surprise-coef", "0.5",
                                  "--surprise-mode", "rnd"])
        self.assertIsNotNone(trainer.rnd)
        self.assertGreaterEqual(rec["surprise"], 0.0)

    def test_pbt_can_select_on_margin(self):
        rec, _ = self._run(["--pbt", "--pbt-every", "1", "--pbt-size", "3",
                            "--pbt-games", "2", "--fitness", "margin"])
        self.assertIn("pbt_best", rec)


class TestABTool(unittest.TestCase):
    def test_paired_report_prefers_the_treatment(self):
        import ab
        tmp = Path(tempfile.mkdtemp())
        try:
            for name, wins in (("a", [0.7] * 8), ("b", [0.5] * 8)):
                d = tmp / name
                d.mkdir()
                with open(d / "metrics.jsonl", "w", encoding="utf-8") as fh:
                    for i, w in enumerate(wins):
                        fh.write(json.dumps({"iter": i + 1,
                                             "holdout_win": w}) + "\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                ab.main(["--a", str(tmp / "a"), "--b", str(tmp / "b")])
            out = buf.getvalue()
            self.assertIn("-> a", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
