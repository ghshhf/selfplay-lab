"""Unit and smoke tests for the neural self-play stack.

Run from the repository root::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.neuro.env import N_ACTIONS, OBS_DIM, NeuralArena
from domains.neuro.opponents import CURRICULUM_LEVELS, heuristic_action
from domains.neuro.train import main
from selfplay.core.curriculum import CurriculumConfig, WorldCurriculum
from selfplay.core.health import HealthConfig, diagnose
from selfplay.core.recorder import RunRecorder
from selfplay.nn.mlp import MLP
from selfplay.nn.ppo import Episode, PPOAgent, PPOConfig, Step


class TestHeuristicExpert(unittest.TestCase):
    def test_distance_is_in_metres(self):
        # obs[12] is dx divided by the arena half width (4m), not by 8m.  A
        # regression here makes the clone expert walk into body contact.
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[12] = 1.6 / 4.0
        self.assertEqual(heuristic_action(obs), 0)  # FWD, out of kick reach
        obs[12] = 1.0 / 4.0
        self.assertEqual(heuristic_action(obs), 3)  # KICK, inside kick reach
        obs[12] = -1.0 / 4.0
        self.assertEqual(heuristic_action(obs), 3)  # symmetric for both sides


class TestWorldCurriculum(unittest.TestCase):
    def _cur(self, **kw):
        return WorldCurriculum(
            CURRICULUM_LEVELS,
            CurriculumConfig(patience=kw.pop("patience", 2),
                             wm_threshold=kw.pop("wm_threshold", 0.02)),
        )

    def test_level_up_needs_both_signals(self):
        cur = self._cur()
        # Saturated but the world model is still surprised -> hold.
        first = cur.observe(0, "saturated", [0.5, 0.4])
        self.assertEqual(first["reason"], "not_mastered")  # patience not met yet
        second = cur.observe(1, "saturated", [0.5, 0.4])
        self.assertEqual(second["level"], 0)
        self.assertEqual(second["action"], "hold")
        self.assertEqual(second["reason"], "world_model_uncertain")
        # Predictable world -> level up.
        step = cur.observe(2, "saturated", [0.01, 0.012])
        self.assertEqual(step["action"], "up")
        self.assertEqual(step["level"], 1)

    def test_patience_and_max_level(self):
        cur = self._cur(patience=3)
        for i in range(2):
            self.assertEqual(cur.observe(i, "saturated", [0.01])["action"], "hold")
        self.assertEqual(cur.observe(2, "saturated", [0.01])["action"], "up")
        for i in range(20):
            cur.observe(3 + i, "saturated", [0.01])
        self.assertEqual(cur.level, cur.max_level)
        self.assertEqual(cur.history[-1]["action"], "hold")
        self.assertEqual(cur.history[-1]["reason"], "max_level")

    def test_floor_lock_steps_down_and_clamps(self):
        cur = self._cur()
        cur.level = 2
        cur.observe(0, "floor_lock", [0.01])
        cur.observe(1, "floor_lock", [0.01])
        self.assertEqual(cur.level, 1)
        cur.observe(2, "floor_lock", [0.01])
        cur.observe(3, "floor_lock", [0.01])
        self.assertEqual(cur.level, 0)
        cur.observe(4, "floor_lock", [0.01])
        cur.observe(5, "floor_lock", [0.01])
        self.assertEqual(cur.level, 0)

    def test_pool_and_checkpoint_roundtrip(self):
        cur = self._cur()
        self.assertEqual(len(cur.scripted_pool()), 1)
        cur.level = 2
        self.assertEqual(len(cur.scripted_pool()), 3)
        saved = cur.state_dict()
        restored = self._cur()
        restored.load_state_dict(saved)
        self.assertEqual(restored.level, 2)
        self.assertEqual(restored.state_dict()["history"], saved["history"])

    def test_health_status_other_resets_streak(self):
        cur = self._cur()
        cur.observe(0, "saturated", [0.01])
        cur.observe(1, "ok", [0.01])
        cur.observe(2, "saturated", [0.01])
        self.assertEqual(cur.level, 0)


class TestMLP(unittest.TestCase):
    def test_backprop_matches_finite_difference(self):
        rng = np.random.default_rng(3)
        net = MLP([3, 5, 2], activation="tanh", rng=rng)
        x = rng.normal(size=(4, 3)).astype(np.float32)
        y = rng.normal(size=(4, 2)).astype(np.float32)
        out = net.forward(x)
        grads = net.backward((out - y) / (x.shape[0] * y.shape[1]))

        eps = 1e-3
        max_err = 0.0
        for li in range(len(net.params)):
            for pi in range(2):
                arr = net.params[li][pi]
                for idx in (tuple(0 for _ in arr.shape), tuple(1 for _ in arr.shape)):
                    old = arr[idx]
                    arr[idx] = old + eps
                    lp = 0.5 * float(np.mean((net.forward(x) - y) ** 2))
                    arr[idx] = old - eps
                    lm = 0.5 * float(np.mean((net.forward(x) - y) ** 2))
                    arr[idx] = old
                    num = (lp - lm) / (2 * eps)
                    max_err = max(max_err, abs(num - float(grads[li][pi][idx])))
        self.assertLess(max_err, 1e-3)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            rng = np.random.default_rng(1)
            net = MLP([4, 3, 2], rng=rng)
            x = rng.normal(size=(2, 4)).astype(np.float32)
            before = net.forward(x).copy()
            net.save(Path(td) / "m.npz")
            loaded = MLP.load(Path(td) / "m.npz")
            after = loaded.forward(x)
            np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


class TestPPO(unittest.TestCase):
    def test_bandit_learns(self):
        agent = PPOAgent(
            obs_dim=1, n_actions=3, hidden=16,
            cfg=PPOConfig(lr=1e-2, epochs=8, batch_size=64, entropy_coef=0.001),
            seed=1,
        )
        rng = np.random.default_rng(2)
        obs = np.array([1.0], dtype=np.float32)
        for _ in range(20):
            episodes = []
            for _ in range(32):
                action, logp, value = agent.act(obs, rng)
                reward = 1.0 if action == 0 else 0.0
                episodes.append(Episode([Step(obs, action, logp, value, reward, True)], obs))
            agent.train(episodes, rng)
        logits = agent.policy.forward(np.array([[1.0]], dtype=np.float32))[0]
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        self.assertGreater(float(p[0]), 0.6)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            agent = PPOAgent(15, 6, hidden=16, seed=4)
            obs = np.zeros(15, dtype=np.float32)
            action, logp, value = agent.act(obs, np.random.default_rng(0))
            agent.save(Path(td) / "a.npz")
            loaded = PPOAgent.load(Path(td) / "a.npz")
            action2, logp2, value2 = loaded.act(obs, np.random.default_rng(0), deterministic=True)
            self.assertEqual(action2, int(np.argmax(agent.policy.forward(np.zeros((1, 15)))[0])))
            np.testing.assert_allclose(logp, logp2, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(value, value2, rtol=1e-5, atol=1e-5)


class TestArena(unittest.TestCase):
    def test_observation_and_episode(self):
        arena = NeuralArena(seed=0, max_frames=120)
        o0, o1 = arena.reset(0)
        self.assertEqual(o0.shape, (OBS_DIM,))
        self.assertEqual(o1.shape, (OBS_DIM,))
        done = False
        info = {}
        for _ in range(200):
            o0, o1, r0, r1, done, info = arena.step(0, 0)
            self.assertEqual(o0.shape, (OBS_DIM,))
            if done:
                break
        self.assertTrue(done)
        self.assertLessEqual(arena.frames, 120)
        self.assertIn("winner", info)


class TestHealthAndRecorder(unittest.TestCase):
    def test_health_pathologies(self):
        floor = diagnose([0.5] * 10, [0.0] * 10)
        self.assertEqual(floor["status"], "floor_lock")
        saturated = diagnose([0.9] * 10, [0.97] * 10)
        self.assertEqual(saturated["status"], "saturated")
        collapse = diagnose([0.5] * 10, [0.5] * 10, entropy=[0.1] * 10)
        self.assertEqual(collapse["status"], "policy_collapse")
        # Deterministic but improving is not a collapse.
        rising = diagnose([0.5] * 10, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.95, 0.95],
                          entropy=[0.1] * 10)
        self.assertNotEqual(rising["status"], "policy_collapse")

    def test_recorder_and_state(self):
        with tempfile.TemporaryDirectory() as td:
            rec = RunRecorder(td)
            rec.log({"iter": 1, "win": 0.5, "note": "a"})
            rec.log({"iter": 2, "win": 0.6, "note": "b"})
            lines = (Path(td) / "progress.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "iter\twin\tnote")
            self.assertEqual(len(lines), 3)
            rec.save_state({"next_iter": 2}, {"p": np.asarray([1.0, 2.0])})
            state, arrays = rec.load_state()
            self.assertEqual(state["next_iter"], 2)
            np.testing.assert_allclose(arrays["p"], [1.0, 2.0])


class TestNeuroTraining(unittest.TestCase):
    def _run(self, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            main(argv)

    def test_smoke_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            base = ["--episodes", "1", "--max-frames", "100",
                    "--eval-episodes", "1", "--warmup", "99"]
            self._run(["--iterations", "2", "--out-dir", td] + base)
            metrics = Path(td) / "metrics.jsonl"
            self.assertTrue(metrics.exists())
            self.assertEqual(len(metrics.read_text().splitlines()), 2)
            self.assertTrue((Path(td) / "manifest.json").exists())
            self.assertTrue((Path(td) / "state.json").exists())
            first = json.loads(metrics.read_text().splitlines()[0])
            self.assertEqual(first["curriculum_level"], 0)
            self.assertIn(first["curriculum_action"], ("hold", "up", "down"))
            self._run(["--resume", "--iterations", "3", "--out-dir", td] + base)
            self.assertEqual(len(metrics.read_text().splitlines()), 3)

    def test_bc_warm_start_is_usable(self):
        # Behaviour cloning from the cheap expert must both fit the demo (loss
        # well below the ln(6) random baseline) and produce a policy that can
        # already win without any PPO update (--epochs 0).
        with tempfile.TemporaryDirectory() as td:
            self._run(["--iterations", "1", "--episodes", "1", "--max-frames", "200",
                       "--eval-episodes", "4", "--epochs", "0", "--warmup", "99",
                       "--bc-episodes", "4", "--bc-epochs", "30", "--out-dir", td])
            rec = json.loads((Path(td) / "metrics.jsonl").read_text().splitlines()[0])
            self.assertIsNotNone(rec["bc_loss"])
            self.assertLess(rec["bc_loss"], float(np.log(N_ACTIONS)))
            self.assertGreaterEqual(rec["random_win"], 0.5)

    def test_resume_is_identical_to_uninterrupted(self):
        base = ["--episodes", "1", "--max-frames", "100",
                "--eval-episodes", "1", "--warmup", "99"]
        with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
            self._run(["--iterations", "3", "--out-dir", da] + base)
            self._run(["--iterations", "2", "--out-dir", db] + base)
            self._run(["--resume", "--iterations", "3", "--out-dir", db] + base)

            def load(p):
                rows = [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
                for row in rows:
                    row.pop("elapsed_sec", None)
                return rows

            a = load(Path(da) / "metrics.jsonl")
            b = load(Path(db) / "metrics.jsonl")
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
