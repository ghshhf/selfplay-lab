"""Neural self-play in the duel arena (NumPy PPO + league + world model).

Examples
--------
Smoke run::

    python domains/neuro/train.py --iterations 2 --episodes 2 --max-frames 300 \
        --out-dir runs/neuro_smoke

Longer run::

    python domains/neuro/train.py --iterations 50 --episodes 8 --max-frames 900 \
        --out-dir runs/neuro

Resume::

    python domains/neuro/train.py --resume --iterations 50 --out-dir runs/neuro
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selfplay.core.curriculum import CurriculumConfig, WorldCurriculum  # noqa: E402
from selfplay.core.health import HealthConfig, diagnose  # noqa: E402
from selfplay.core.manifest import RunManifest, make_run_id  # noqa: E402
from selfplay.core.recorder import RunRecorder  # noqa: E402
from selfplay.nn.ppo import (Episode, PPOAgent, PPOConfig, Step,  # noqa: E402
                             log_softmax)
from selfplay.nn.world_model import WorldModel  # noqa: E402

try:
    from .env import (ARTICULATED_OBS_DIM, ArticulatedArena, N_ACTIONS,
                      NeuralArena, OBS_DIM)  # noqa: E402
    from .opponents import (CURRICULUM_LEVELS, RandomOpponent,  # noqa: E402
                            SCRIPTED_GENES, ScriptedOpponent,
                            SnapshotOpponent, heuristic_action)
except ImportError:  # running as a script: python domains/neuro/train.py
    from env import (ARTICULATED_OBS_DIM, ArticulatedArena, N_ACTIONS,
                     NeuralArena, OBS_DIM)  # noqa: E402
    from opponents import (CURRICULUM_LEVELS, RandomOpponent,  # noqa: E402
                           SCRIPTED_GENES, ScriptedOpponent,
                           SnapshotOpponent, heuristic_action)


class NeuroTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(args.seed)
        self.next_iter = 0

        if args.env == "articulated":
            self.arena_cls = ArticulatedArena
            self.obs_dim = ARTICULATED_OBS_DIM
            self.reach_scale = 0.86   # limb geometry is shorter than the 1-D reach
        else:
            self.arena_cls = NeuralArena
            self.obs_dim = OBS_DIM
            self.reach_scale = 1.0

        self.ppo_cfg = PPOConfig(
            gamma=args.gamma, lam=args.lam, clip=args.clip,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            entropy_coef=args.entropy_coef, value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
        )
        self.agent = PPOAgent(self.obs_dim, N_ACTIONS, hidden=args.hidden,
                              cfg=self.ppo_cfg, seed=args.seed)
        self.world_model = None
        if args.world_model:
            self.world_model = WorldModel(self.obs_dim, N_ACTIONS,
                                          hidden=args.hidden,
                                          lr=args.world_lr, seed=args.seed + 1)

        self.league: list[PPOAgent] = []
        self.history: list[float] = []
        self.holdout_history: list[float] = []
        self.entropy_history: list[float] = []
        self.world_mse_history: list[float] = []
        self.best_holdout = -1.0
        self.bc_loss = None
        self.run_id = ""

        self.curriculum = None
        if args.curriculum:
            self.curriculum = WorldCurriculum(
                CURRICULUM_LEVELS,
                CurriculumConfig(patience=args.curriculum_patience,
                                 wm_threshold=args.wm_level_threshold),
            )

        self.recorder = RunRecorder(self.out_dir, resume=args.resume)
        self._prepare_manifest()
        if args.resume:
            self._load_state()
        elif args.bc_episodes > 0:
            # Bootstrap the neural policy from a cheap scripted expert.  PPO
            # then fine-tunes it, instead of asking a randomly initialised
            # network to discover "walk close, then kick" from sparse rewards.
            self.bc_loss = self._behavior_clone()

    # --------------------------------------------------------------- manifest
    def _prepare_manifest(self):
        path = self.out_dir / "manifest.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.run_id = data.get("run_id", "")
            if not self.args.resume and (self.out_dir / "metrics.jsonl").exists():
                raise SystemExit(
                    "output dir already contains a run; use --resume or a new --out-dir"
                )
            return
        self.run_id = make_run_id("neuro", self.args.seed)
        manifest = RunManifest(
            run_id=self.run_id, seed=self.args.seed, config=vars(self.args),
            command=" ".join(sys.argv), repo_root=str(ROOT),
        )
        manifest.write(path)

    # ----------------------------------------------------------- checkpoints
    def _arrays(self):
        arrays = dict(self.agent.state_dict("agent"))
        if self.world_model is not None:
            arrays.update(self.world_model.state_dict("world"))
        for i, snap in enumerate(self.league):
            arrays.update(snap.state_dict("league%d" % i))
        return arrays

    def _save_state(self):
        state = {
            "next_iter": self.next_iter,
            "run_id": self.run_id,
            "rng_state": self.rng.bit_generator.state,
            "history": self.history,
            "holdout_history": self.holdout_history,
            "entropy_history": self.entropy_history,
            "world_mse_history": self.world_mse_history,
            "best_holdout": self.best_holdout,
            "bc_loss": self.bc_loss,
            "league_size": len(self.league),
            "curriculum": self.curriculum.state_dict() if self.curriculum else None,
        }
        self.recorder.save_state(state, self._arrays())

    def _load_state(self):
        state, arrays = self.recorder.load_state()
        if not state:
            raise SystemExit("--resume requested but no state.json was found")
        self.next_iter = int(state.get("next_iter", 0))
        self.history = list(state.get("history", []))
        self.holdout_history = list(state.get("holdout_history", []))
        self.entropy_history = list(state.get("entropy_history", []))
        self.world_mse_history = list(state.get("world_mse_history", []))
        self.best_holdout = float(state.get("best_holdout", -1.0))
        self.bc_loss = state.get("bc_loss")
        cur_state = state.get("curriculum")
        if self.curriculum is not None and cur_state:
            self.curriculum.load_state_dict(cur_state)
        if "rng_state" in state:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = state["rng_state"]
        self.agent.load_state_dict(arrays, "agent")
        if self.world_model is not None and "world__meta" in arrays:
            self.world_model.load_state_dict(arrays, "world")

        self.league = []
        i = 0
        while ("league%d__meta" % i) in arrays:
            snap = PPOAgent(self.obs_dim, N_ACTIONS, hidden=self.args.hidden,
                            cfg=self.ppo_cfg, seed=0)
            snap.load_state_dict(arrays, "league%d" % i)
            self.league.append(snap)
            i += 1

    # ------------------------------------------------------- imitation warm-up
    def _behavior_clone(self):
        gap = self.args.train_gap
        pocket = self.args.env == "articulated"
        obs, acts = [], []
        demo_rng = np.random.default_rng(self.args.seed + 100003)
        # Vary who the expert fights.  Cloning a mirror match teaches the net
        # to read the opponent's pose instead of the distance, which then
        # collapses against a different scripted style.
        pool = [
            RandomOpponent(),
            ScriptedOpponent(SCRIPTED_GENES[0], reach_scale=self.reach_scale),
            ScriptedOpponent(SCRIPTED_GENES[1], reach_scale=self.reach_scale),
        ]
        for i in range(self.args.bc_episodes):
            seed = int(demo_rng.integers(2 ** 31))
            kind = i % (len(pool) + 1)
            for side in (0, 1):
                arena = self.arena_cls(seed=seed, max_frames=self.args.max_frames,
                                       start_gap=gap)
                o0, o1 = arena.reset(seed)
                opp = None
                if kind < len(pool):
                    opp = pool[kind]
                    opp.reset(int(demo_rng.integers(2 ** 31)))
                while True:
                    if side == 0:
                        a0 = int(heuristic_action(o0, pocket=pocket))
                        a1 = (opp.act(arena.env.p1, arena.env.p0, o1) if opp
                              else int(heuristic_action(o1, pocket=pocket)))
                        obs.append(o0)
                        acts.append(a0)
                    else:
                        a0 = (opp.act(arena.env.p0, arena.env.p1, o0) if opp
                              else int(heuristic_action(o0, pocket=pocket)))
                        a1 = int(heuristic_action(o1, pocket=pocket))
                        obs.append(o1)
                        acts.append(a1)
                    o0, o1, _r0, _r1, done, _info = arena.step(a0, a1)
                    if done:
                        break
        if not obs:
            return None
        x = np.stack(obs).astype(np.float32)
        y = np.asarray(acts, dtype=np.int64)
        onehot_cache = np.eye(self.agent.n_actions, dtype=np.float64)
        epoch_loss = 0.0
        for _epoch in range(self.args.bc_epochs):
            order = self.rng.permutation(len(x))
            batch_losses = []
            for start in range(0, len(x), self.args.batch_size):
                idx = order[start:start + self.args.batch_size]
                if len(idx) < 2:
                    continue
                xb, yb = x[idx], y[idx]
                bsz = len(idx)
                logits = self.agent.policy.forward(xb)
                lp = log_softmax(logits)
                p = np.exp(lp)
                batch_losses.append(-float(np.mean(lp[np.arange(bsz), yb])))
                dlogits = (p - onehot_cache[yb]) / bsz
                grads = self.agent.policy.backward(dlogits.astype(np.float32))
                self.agent.policy_adam.step(grads, lr=self.args.bc_lr,
                                            max_norm=self.args.max_grad_norm)
            if batch_losses:
                epoch_loss = float(np.mean(batch_losses))
        return round(epoch_loss, 4)

    # ------------------------------------------------------------- opponents
    def _scripted_pool(self):
        if self.curriculum is not None:
            return self.curriculum.scripted_pool()
        return SCRIPTED_GENES

    def _holdout_gene(self):
        if self.curriculum is not None:
            return self.curriculum.holdout_gene()
        return SCRIPTED_GENES[1]

    def _sample_opponent(self, iteration: int, k: int):
        if iteration < self.args.warmup:
            return RandomOpponent()
        r = float(self.rng.random())
        if self.league and r < 0.45:
            idx = int(self.rng.integers(len(self.league)))
            return SnapshotOpponent(self.league[idx])
        if r < 0.80:
            pool = self._scripted_pool()
            gene = pool[int(self.rng.integers(len(pool)))]
            return ScriptedOpponent(gene, reach_scale=self.reach_scale)
        return RandomOpponent()

    # --------------------------------------------------------------- episodes
    def _play(self, opponent, seed: int, collect: bool):
        arena = self.arena_cls(seed=seed, max_frames=self.args.max_frames,
                            start_gap=self.args.train_gap)
        o0, o1 = arena.reset(seed)
        side = int(self.rng.integers(2))
        ep = Episode()
        total_reward = 0.0
        info = {"winner": -1}
        while True:
            me_obs = o0 if side == 0 else o1
            action, logp, value = self.agent.act(me_obs, self.rng)

            if side == 0:
                opp_action = opponent.act(arena.env.p1, arena.env.p0, o1)
                a0, a1 = action, opp_action
            else:
                opp_action = opponent.act(arena.env.p0, arena.env.p1, o0)
                a0, a1 = opp_action, action

            n0, n1, r0, r1, done, info = arena.step(a0, a1)
            reward = r0 if side == 0 else r1
            total_reward += reward
            if collect:
                ep.steps.append(Step(
                    obs=np.asarray(me_obs, dtype=np.float32).copy(),
                    action=action, logp=logp, value=value,
                    reward=reward * self.args.reward_scale, done=done,
                ))
            o0, o1 = n0, n1
            if done:
                break

        ep.last_obs = np.asarray(o0 if side == 0 else o1, dtype=np.float32)
        winner = int(info.get("winner", -1))
        if winner < 0:
            outcome = 0.5
        elif winner == side:
            outcome = 1.0
        else:
            outcome = 0.0
        return ep, {
            "outcome": outcome,
            "frames": arena.frames,
            "return": total_reward,
            "winner": winner,
            "clashes": int(info.get("clashes", 0)),
            "head_hits": int(info.get("head_hits", 0)),
        }

    def _collect(self, n: int, iteration: int):
        episodes = []
        outcomes = []
        frames = []
        returns = []
        clashes = []
        heads = []
        for k in range(n):
            opp = self._sample_opponent(iteration, k)
            seed = int(self.rng.integers(2 ** 31))
            opp.reset(int(self.rng.integers(2 ** 31)))
            ep, stat = self._play(opp, seed, collect=True)
            episodes.append(ep)
            outcomes.append(stat["outcome"])
            frames.append(stat["frames"])
            returns.append(stat["return"])
            clashes.append(stat["clashes"])
            heads.append(stat["head_hits"])
        return episodes, {
            "win": float(np.mean(outcomes)) if outcomes else 0.0,
            "frames": float(np.mean(frames)) if frames else 0.0,
            "return": float(np.mean(returns)) if returns else 0.0,
            "clashes": float(np.mean(clashes)) if clashes else 0.0,
            "head_hits": float(np.mean(heads)) if heads else 0.0,
            "steps": int(sum(len(ep.steps) for ep in episodes)),
        }

    def _evaluate(self, n: int, kind: str):
        # A fixed seed suite, independent of the training RNG, so the same
        # checkpoint is always scored on the same matches.
        outcomes = []
        frames = []
        clashes = []
        heads = []
        eval_rng = np.random.default_rng(0)
        for seed in range(2000, 2000 + n):
            for side in (0, 1):
                if kind == "random":
                    opp = RandomOpponent()
                else:
                    opp = ScriptedOpponent(self._holdout_gene(),
                                          reach_scale=self.reach_scale)
                opp.reset(seed * 2 + side)
                arena = self.arena_cls(seed=seed, max_frames=self.args.max_frames,
                                    start_gap=self.args.eval_gap)
                o0, o1 = arena.reset(seed)
                info = {"winner": -1}
                while True:
                    me_obs = o0 if side == 0 else o1
                    action, _logp, _value = self.agent.act(me_obs, eval_rng, deterministic=True)
                    if side == 0:
                        opp_action = opp.act(arena.env.p1, arena.env.p0, o1)
                        a0, a1 = action, opp_action
                    else:
                        opp_action = opp.act(arena.env.p0, arena.env.p1, o0)
                        a0, a1 = opp_action, action
                    n0, n1, _r0, _r1, done, info = arena.step(a0, a1)
                    o0, o1 = n0, n1
                    if done:
                        break
                w = int(info.get("winner", -1))
                outcomes.append(0.5 if w < 0 else (1.0 if w == side else 0.0))
                frames.append(arena.frames)
                clashes.append(int(info.get("clashes", 0)))
                heads.append(int(info.get("head_hits", 0)))
        return {
            "win": float(np.mean(outcomes)) if outcomes else 0.0,
            "frames": float(np.mean(frames)) if frames else 0.0,
            "clashes": float(np.mean(clashes)) if clashes else 0.0,
            "head_hits": float(np.mean(heads)) if heads else 0.0,
            "games": len(outcomes),
        }

    # --------------------------------------------------------------- training
    def _world_model_step(self, episodes):
        if self.world_model is None:
            return None
        obs, actions, next_obs = [], [], []
        for ep in episodes:
            for i, step in enumerate(ep.steps):
                nxt = ep.steps[i + 1].obs if i + 1 < len(ep.steps) else ep.last_obs
                if nxt is None:
                    continue
                obs.append(step.obs)
                actions.append(step.action)
                next_obs.append(nxt)
        if not obs:
            return None
        return self.world_model.train(
            np.stack(obs), np.asarray(actions, dtype=np.int64), np.stack(next_obs),
            self.rng, epochs=self.args.world_epochs,
        )

    def _entropy(self, episodes):
        obs = []
        for ep in episodes:
            for step in ep.steps[:32]:
                obs.append(step.obs)
        if not obs:
            return 0.0
        return float(np.mean([self.agent.entropy_of(o) for o in obs[:128]]))

    def _snapshot(self):
        snap = PPOAgent(self.obs_dim, N_ACTIONS, hidden=self.args.hidden,
                        cfg=self.ppo_cfg, seed=0)
        snap.load_state_dict(self.agent.state_dict(), "agent")
        self.league.append(snap)
        if len(self.league) > self.args.league_size:
            self.league.pop(0)

    def train_iteration(self, iteration: int):
        episodes, stats = self._collect(self.args.episodes, iteration)
        ppo = self.agent.train(episodes, self.rng)
        world_mse = self._world_model_step(episodes)
        if world_mse is not None:
            self.world_mse_history.append(float(world_mse))
            self.world_mse_history = self.world_mse_history[-50:]

        holdout = self._evaluate(self.args.eval_episodes, "holdout")
        random_eval = self._evaluate(self.args.eval_episodes, "random")
        entropy = self._entropy(episodes)

        self.history.append(float(stats["win"]))
        self.history = self.history[-50:]
        self.holdout_history.append(float(holdout["win"]))
        self.holdout_history = self.holdout_history[-50:]
        self.entropy_history.append(entropy)
        self.entropy_history = self.entropy_history[-50:]
        self.best_holdout = max(self.best_holdout, float(holdout["win"]))

        health = diagnose(
            self.history, self.holdout_history, self.entropy_history,
            self.world_mse_history,
            HealthConfig(window=self.args.health_window),
        )

        curriculum_step = None
        if self.curriculum is not None:
            curriculum_step = self.curriculum.observe(
                iteration, health["status"], self.world_mse_history,
            )

        if (iteration + 1) % self.args.snapshot_every == 0:
            self._snapshot()

        rec = {
            "iter": iteration + 1,
            "episodes": self.args.episodes,
            "steps": stats["steps"],
            "train_win": round(float(stats["win"]), 4),
            "train_return": round(float(stats["return"]), 2),
            "avg_frames": round(float(stats["frames"]), 1),
            "clashes": round(float(stats["clashes"]), 2),
            "eval_clashes": round(float(holdout["clashes"]), 2),
            "head_hits": round(float(stats["head_hits"]), 2),
            "eval_head_hits": round(float(holdout["head_hits"]), 2),
            "holdout_win": round(float(holdout["win"]), 4),
            "holdout_frames": round(float(holdout["frames"]), 1),
            "random_win": round(float(random_eval["win"]), 4),
            "entropy": round(float(entropy), 4),
            "world_mse": None if world_mse is None else round(float(world_mse), 5),
            "policy_loss": round(float(ppo["policy_loss"]), 5),
            "value_loss": round(float(ppo["value_loss"]), 5),
            "approx_kl": round(float(ppo["approx_kl"]), 5),
            "clip_frac": round(float(ppo["clip_frac"]), 4),
            "grad_norm": round(float(ppo["grad_norm"]), 4),
            "param_norm": round(float(self.agent.param_norm()), 4),
            "bc_loss": self.bc_loss,
            "league_size": len(self.league),
            "curriculum_level": None if self.curriculum is None else self.curriculum.level,
            "curriculum_name": None if self.curriculum is None else self.curriculum.level_name,
            "curriculum_action": None if curriculum_step is None else curriculum_step["action"],
            "curriculum_reason": None if curriculum_step is None else curriculum_step["reason"],
            "health": health["status"],
            "health_reasons": health["reasons"],
        }
        return rec

    def run(self):
        for iteration in range(self.next_iter, self.args.iterations):
            t0 = time.time()
            rec = self.train_iteration(iteration)
            rec["elapsed_sec"] = round(time.time() - t0, 2)
            self.recorder.log(rec)
            self.next_iter = iteration + 1
            self._save_state()
            print(json.dumps(rec, ensure_ascii=False), flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description="Neural self-play in the duel arena")
    ap.add_argument("--env", choices=("duel", "articulated"), default="duel",
                    help="duel: 1-D reach check; articulated: limb-capsule contact")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="runs/neuro")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--reward-scale", type=float, default=0.01)
    ap.add_argument("--world-model", action="store_true", default=True)
    ap.add_argument("--no-world-model", dest="world_model", action="store_false")
    ap.add_argument("--world-lr", type=float, default=1e-3)
    ap.add_argument("--world-epochs", type=int, default=2)
    ap.add_argument("--snapshot-every", type=int, default=2)
    ap.add_argument("--league-size", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--health-window", type=int, default=10)
    ap.add_argument("--train-gap", type=float, default=2.2)
    ap.add_argument("--eval-gap", type=float, default=2.2)
    ap.add_argument("--bc-episodes", type=int, default=8)
    ap.add_argument("--bc-epochs", type=int, default=60)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--curriculum", action="store_true", default=True)
    ap.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    ap.add_argument("--curriculum-patience", type=int, default=2)
    ap.add_argument("--wm-level-threshold", type=float, default=0.02)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    trainer = NeuroTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
