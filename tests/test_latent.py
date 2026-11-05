"""Tests for the latent world model (encoder + predictor + decoder)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selfplay.nn.latent import LatentWorldModel


def make_data(n=96, obs_dim=6, n_actions=3, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, obs_dim)).astype(np.float32)
    actions = rng.integers(0, n_actions, size=n)
    # Deterministic next state so a correct model can drive the error down.
    nxt = np.roll(obs, -1, axis=1)
    nxt[:, -1] = actions / float(n_actions)
    return obs, actions, nxt


class TestLatentModel(unittest.TestCase):
    def test_bottleneck_shapes(self):
        m = LatentWorldModel(6, 3, latent=4, hidden=16, seed=0)
        z = m.encode(np.zeros((5, 6), dtype=np.float32))
        self.assertEqual(z.shape, (5, 4))
        pred = m.predict(np.zeros((5, 6), dtype=np.float32), np.zeros(5, dtype=np.int64))
        self.assertEqual(pred.shape, (5, 4))

    def test_training_reduces_latent_error(self):
        obs, actions, nxt = make_data()
        m = LatentWorldModel(6, 3, latent=4, hidden=16, seed=0)
        rng = np.random.default_rng(1)
        first = m.train(obs, actions, nxt, rng, epochs=2)
        for _ in range(12):
            last = m.train(obs, actions, nxt, rng, epochs=2)
        self.assertGreaterEqual(first["latent_mse"], 0.0)
        self.assertLess(last["latent_mse"], first["latent_mse"])
        self.assertLess(last["recon_mse"], first["recon_mse"])

    def test_bottleneck_is_smaller_than_observation(self):
        m = LatentWorldModel(33, 9, latent=8, hidden=32, seed=0)
        self.assertLess(m.latent_dim, m.obs_dim)

    def test_state_roundtrip_and_shape_guard(self):
        obs, actions, nxt = make_data()
        m = LatentWorldModel(6, 3, latent=4, hidden=16, seed=0)
        m.train(obs, actions, nxt, np.random.default_rng(2), epochs=2)
        state = m.state_dict("world")
        clone = LatentWorldModel(6, 3, latent=4, hidden=16, seed=0)
        clone.load_state_dict(state, "world")
        np.testing.assert_allclose(
            m.encode(obs[:4]), clone.encode(obs[:4]), rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            LatentWorldModel(6, 3, latent=2, hidden=16, seed=0).load_state_dict(
                state, "world")

    def test_error_matches_manual_mse(self):
        obs, actions, nxt = make_data()
        m = LatentWorldModel(6, 3, latent=4, hidden=16, seed=0)
        e = m.error(obs, actions, nxt)
        pred = m.predict(obs, actions)
        target = m.encode(nxt)
        self.assertAlmostEqual(e, float(np.mean((pred - target) ** 2)), places=5)


if __name__ == "__main__":
    unittest.main()
