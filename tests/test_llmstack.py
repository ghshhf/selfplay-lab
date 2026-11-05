"""Tests for the LLM-style training stack."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selfplay.nn.llmstack import (EMA, EntropySchedule, LRSchedule,  # noqa: E402
                                  RejectionBuffer, mirror_batch, mirror_obs)
from selfplay.nn.mlp import Adam, MLP  # noqa: E402
from selfplay.nn.ppo import Episode, Step  # noqa: E402


def _episode(actions, reward=1.0):
    ep = Episode()
    for a in actions:
        ep.steps.append(Step(obs=np.zeros(3, dtype=np.float32), action=int(a),
                             logp=0.0, value=0.0, reward=reward, done=False))
    return ep


class TestLRSchedule(unittest.TestCase):
    def test_warmup_then_cosine_floor(self):
        s = LRSchedule(base_lr=1e-3, warmup_iters=2, total_iters=10, min_ratio=0.1)
        self.assertAlmostEqual(s.at(0), 1e-3 * 0.5)
        self.assertAlmostEqual(s.at(1), 1e-3)
        self.assertAlmostEqual(s.at(10), 1e-3 * 0.1, places=9)
        values = [s.at(i) for i in range(11)]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(values[1:], values[2:])))

    def test_flat_when_min_ratio_is_one(self):
        s = LRSchedule(1e-3, warmup_iters=0, total_iters=5, min_ratio=1.0)
        for i in range(6):
            self.assertAlmostEqual(s.at(i), 1e-3)


class TestEntropySchedule(unittest.TestCase):
    def test_anneals_to_end(self):
        s = EntropySchedule(0.01, 0.003, total_iters=4)
        self.assertAlmostEqual(s.at(0), 0.01)
        self.assertAlmostEqual(s.at(4), 0.003)

    def test_none_keeps_constant(self):
        s = EntropySchedule(0.01, None, total_iters=3)
        self.assertAlmostEqual(s.at(3), 0.01)


class TestEMA(unittest.TestCase):
    def test_first_update_copies_then_averages(self):
        mlp = MLP([2, 2, 1], rng=np.random.default_rng(0))
        ema = EMA([mlp], decay=0.9)
        ema.update()
        self.assertTrue(np.allclose(ema.shadow[0][0][0], mlp.params[0][0]))
        for layer in mlp.params:
            for p in layer:
                p[...] = 0.0
        for _ in range(50):
            ema.update()
        self.assertTrue(np.allclose(ema.shadow[0][0][0], 0.0, atol=0.02))

    def test_copy_to_restores_shadow(self):
        a = MLP([3, 2, 1], rng=np.random.default_rng(1))
        b = MLP([3, 2, 1], rng=np.random.default_rng(1))
        ema = EMA([a], decay=0.5)
        ema.update()
        for layer in a.params:
            for p in layer:
                p[...] = 7.0
        ema.update()
        ema.copy_to([b])
        self.assertFalse(np.allclose(a.params[0][0], b.params[0][0]))
        self.assertTrue(np.allclose(b.params[0][0], ema.shadow[0][0][0]))

    def test_state_round_trip(self):
        mlp = MLP([2, 2, 1], rng=np.random.default_rng(2))
        ema = EMA([mlp], decay=0.9)
        ema.update()
        other = EMA([MLP([2, 2, 1], rng=np.random.default_rng(3))], decay=0.9)
        other.load_state_dict(ema.state_dict(), "ema")
        self.assertEqual(other.updates, ema.updates)
        self.assertTrue(np.allclose(other.shadow[0][0][0], ema.shadow[0][0][0]))


class TestRejectionBuffer(unittest.TestCase):
    def test_keeps_only_top_fraction(self):
        eps = [_episode([0, 1, 2]), _episode([3, 4, 5, 6]),
               _episode([7]), _episode([8, 9])]
        buf = RejectionBuffer(obs_dim=3, max_steps=100)
        buf.add(eps, returns=[1.0, 5.0, 2.0, 9.0], keep_frac=0.25)
        # top 25% of 4 episodes = 1 -> the one with return 9.0 (2 steps)
        self.assertEqual(len(buf), 2)
        _x, y = buf.arrays()
        self.assertEqual(y.tolist(), [8, 9])

    def test_caps_oldest(self):
        buf = RejectionBuffer(obs_dim=3, max_steps=5)
        for _ in range(4):
            buf.add([_episode([0, 0, 0, 0])], returns=[1.0], keep_frac=1.0)
        self.assertEqual(len(buf), 5)
        self.assertEqual(buf.acts[-1], 0)


class TestMirror(unittest.TestCase):
    def test_involution_on_freeform_obs(self):
        obs = np.arange(33, dtype=np.float32) / 10.0
        once = mirror_obs(obs, 33)
        twice = mirror_obs(once, 33)
        self.assertTrue(np.allclose(twice, obs))
        # x, vx, facing, opp x/vx/facing, dx and limb offsets flip
        for i in (0, 1, 5, 6, 7, 11, 12, 15, 17, 19, 21):
            self.assertAlmostEqual(once[i], -obs[i])
        # heights, hp, timers, chakra, cooldowns do not
        for i in (2, 3, 4, 8, 9, 10, 13, 16, 18, 20, 22, 23, 24, 25):
            self.assertAlmostEqual(once[i], obs[i])

    def test_batch_matches_single(self):
        x = np.arange(3 * 23, dtype=np.float32).reshape(3, 23)
        batch = mirror_batch(x, 23)
        for r in range(3):
            self.assertTrue(np.allclose(batch[r], mirror_obs(x[r], 23)))

    def test_15_dim_mapping_leaves_cooldown_alone(self):
        obs = np.ones(15, dtype=np.float32)
        out = mirror_obs(obs, 15)
        self.assertEqual(len(out), 15)
        self.assertAlmostEqual(out[15 - 1], obs[15 - 1])  # cooldown unchanged


class TestAdamW(unittest.TestCase):
    def test_weight_decay_shrinks_weights_without_touching_bias(self):
        mlp = MLP([2, 2, 1], rng=np.random.default_rng(0))
        for layer in mlp.params:
            for p in layer:
                p[...] = 1.0
        opt = Adam(mlp.params, lr=0.01, weight_decay=0.1)
        grads = [[np.zeros_like(p) for p in layer] for layer in mlp.params]
        opt.step(grads)
        # w <- w - lr*wd*w ; bias untouched by weight decay (but Adam on zero
        # gradient leaves it where it was)
        self.assertAlmostEqual(float(mlp.params[0][0][0, 0]), 0.999, places=4)
        self.assertAlmostEqual(float(mlp.params[0][1][0]), 1.0, places=6)

    def test_zero_decay_leaves_weights_alone(self):
        mlp = MLP([2, 2, 1], rng=np.random.default_rng(0))
        before = [p.copy() for layer in mlp.params for p in layer]
        opt = Adam(mlp.params, lr=0.01, weight_decay=0.0)
        grads = [[np.zeros_like(p) for p in layer] for layer in mlp.params]
        opt.step(grads)
        after = [p for layer in mlp.params for p in layer]
        for a, b in zip(before, after):
            self.assertTrue(np.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
