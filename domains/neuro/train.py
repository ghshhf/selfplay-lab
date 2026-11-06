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

from selfplay.core.budget import ComputeLedger  # noqa: E402
from selfplay.core.curriculum import CurriculumConfig, WorldCurriculum  # noqa: E402
from selfplay.core.health import HealthConfig, diagnose  # noqa: E402
from selfplay.core.manifest import RunManifest, make_run_id  # noqa: E402
from selfplay.core.recorder import RunRecorder  # noqa: E402
from selfplay.nn.ppo import (Episode, PPOAgent, PPOConfig, Step,  # noqa: E402
                             log_softmax)
from selfplay.core.stats import summarise  # noqa: E402
from selfplay.nn.latent import LatentWorldModel  # noqa: E402
from selfplay.nn.llmstack import (EMA, EntropySchedule, LRSchedule,  # noqa: E402
                                  RejectionBuffer, mirror_batch)
from selfplay.nn.pbt import PBTPopulation, clone_agent  # noqa: E402
from selfplay.nn.rnd import RND  # noqa: E402
from selfplay.nn.world_model import WorldModel  # noqa: E402

AFFECT_KEYS = ("composure", "fear", "anger", "momentum")

try:
    from .env import (ACTION_NAMES, ARTICULATED_OBS_DIM, ArticulatedArena,
                      FREEFORM_ACTION_NAMES, FREEFORM_ACTION_NAMES_3D,
                      FREEFORM_N_ACTIONS, FREEFORM_N_ACTIONS_3D,
                      FREEFORM_OBS_DIM, FREEFORM_OBS_DIM_3D,
                      FREEFORM_EMOTION_EXTRA,
                      FreeformArena, N_ACTIONS,
                      NeuralArena, OBS_DIM)  # noqa: E402
    from .opponents import (CURRICULUM_LEVELS, RandomOpponent,  # noqa: E402
                            SCRIPTED_GENES, ScriptedOpponent,
                            SnapshotOpponent, heuristic_action)
    from .selfevolve import SelfPlayPool, mutate_gene, random_gene  # noqa: E402
except ImportError:  # running as a script: python domains/neuro/train.py
    from env import (ACTION_NAMES, ARTICULATED_OBS_DIM, ArticulatedArena,
                     FREEFORM_ACTION_NAMES, FREEFORM_ACTION_NAMES_3D,
                     FREEFORM_N_ACTIONS, FREEFORM_N_ACTIONS_3D,
                     FREEFORM_OBS_DIM, FREEFORM_OBS_DIM_3D,
                     FREEFORM_EMOTION_EXTRA,
                     FreeformArena, N_ACTIONS,
                     NeuralArena, OBS_DIM)  # noqa: E402
    from opponents import (CURRICULUM_LEVELS, RandomOpponent,  # noqa: E402
                           SCRIPTED_GENES, ScriptedOpponent,
                           SnapshotOpponent, heuristic_action)
    from selfevolve import SelfPlayPool, mutate_gene, random_gene  # noqa: E402


def _apply_selfevolve(args):
    """Zero-data mode: no demos, no authored ladder, no KL anchor.

    Runs before the LLM preset, so the values set here are explicit and the
    preset will not overwrite them.
    """
    if getattr(args, "selfevolve", False):
        if args.bc_episodes is None:
            args.bc_episodes = 0
        if args.kl_coef is None:
            args.kl_coef = 0.0
    elif args.bc_episodes is None:
        args.bc_episodes = 8


def _apply_llm_stack(args):
    """Turn on the whole LLM training recipe with one flag.

    Every knob defaults to ``None`` so an explicit ``0`` means "off" rather
    than "unspecified".  That is what makes ablation runs trustworthy -- a
    preset must never silently override ``--kl-coef 0``.
    """
    preset = {
        "weight_decay": 0.01,
        "kl_coef": 0.02,
        "lr_warmup": min(3, max(1, args.iterations // 4)),
        "lr_min_ratio": 0.1,
        "ema_decay": 0.9,
        "rft_frac": 0.25,
        "rft_epochs": 1,
        "entropy_final": args.entropy_coef * 0.3,
    }
    off = {"weight_decay": 0.0, "kl_coef": 0.0, "lr_warmup": 0,
           "lr_min_ratio": 1.0, "ema_decay": 0.0, "rft_frac": 0.0,
           "rft_epochs": 0, "entropy_final": None}
    if getattr(args, "llm_stack", False):
        preset["mirror_bc"] = True
    for name, value in (preset if args.llm_stack else {}).items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    for name, value in off.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    if args.llm_stack and not args.mirror_bc:
        args.mirror_bc = True


class NeuroTrainer:
    def __init__(self, args: argparse.Namespace):
        _apply_selfevolve(args)
        _apply_llm_stack(args)
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(args.seed)
        self.next_iter = 0

        if args.env == "articulated":
            self.arena_cls = ArticulatedArena
            self.obs_dim = ARTICULATED_OBS_DIM
            self.n_actions = N_ACTIONS
            self.reach_scale = 0.86   # limb geometry is shorter than the 1-D reach
        elif args.env == "freeform":
            self.arena_cls = FreeformArena
            self.obs_dim = ((FREEFORM_OBS_DIM_3D if args.arena3d
                             else FREEFORM_OBS_DIM)
                            + (FREEFORM_EMOTION_EXTRA if args.emotion else 0))
            self.n_actions = (FREEFORM_N_ACTIONS_3D if args.arena3d
                              else FREEFORM_N_ACTIONS)
            self.reach_scale = 0.86
        else:
            self.arena_cls = NeuralArena
            self.obs_dim = OBS_DIM
            self.n_actions = N_ACTIONS
            self.reach_scale = 1.0

        # Weight classes are part of the match-up, not the policy: both seats
        # still share one network, but one body is heavier.  Mass only has an
        # effect with --physics, so it is ignored otherwise.
        self.env_cfg = None
        if args.env == "freeform" and args.physics:
            m0, m1 = float(args.mass0), float(args.mass1)
            if m0 != 70.0 or m1 != 70.0:
                self.env_cfg = {"mass0": m0, "mass1": m1}

        self.policy_dim = (args.latent_dim if args.latent_policy
                           else self.obs_dim)
        self.encoder = None          # set when --latent-policy is active

        self.ppo_cfg = PPOConfig(
            gamma=args.gamma, lam=args.lam, clip=args.clip,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            entropy_coef=args.entropy_coef, value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            weight_decay=args.weight_decay, kl_coef=args.kl_coef,
        )
        self.agent = PPOAgent(self.policy_dim, self.n_actions, hidden=args.hidden,
                              cfg=self.ppo_cfg, seed=args.seed)
        self.world_model = None
        if args.world_model or args.latent_policy:
            if getattr(args, "latent_world", False) or args.latent_policy:
                self.world_model = LatentWorldModel(
                    self.obs_dim, self.n_actions, hidden=args.hidden,
                    latent=args.latent_dim, lr=args.world_lr,
                    seed=args.seed + 1)
            else:
                self.world_model = WorldModel(self.obs_dim, self.n_actions,
                                              hidden=args.hidden,
                                              lr=args.world_lr,
                                              seed=args.seed + 1)

        # Curiosity has two possible sources.  RND does not need the world
        # model at all, so it is independent of `--world-model`.
        self.rnd = None
        if args.surprise_mode == "rnd" and args.surprise_coef > 0:
            self.rnd = RND(self.obs_dim, hidden=args.hidden,
                           seed=args.seed + 2)

        self.league: list[PPOAgent] = []
        self.history: list[float] = []
        self.holdout_history: list[float] = []
        self.entropy_history: list[float] = []
        self.world_mse_history: list[float] = []
        self.best_holdout = -1.0
        self.bc_loss = None
        self.recon_mse = None
        self.encoder_metrics = None
        self._surprise_ema = 1e-3
        self._rnd_ema = 1e-3
        self._surprise_sum = 0.0
        self.run_id = ""
        self.ledger = ComputeLedger()
        self._last_eval = None

        self.curriculum = None
        if args.curriculum:
            self.curriculum = WorldCurriculum(
                CURRICULUM_LEVELS,
                CurriculumConfig(patience=args.curriculum_patience,
                                 wm_threshold=args.wm_level_threshold),
            )

        self.recorder = RunRecorder(self.out_dir, resume=args.resume)
        self._prepare_manifest()

        # ---- LLM-style training stack (all optional, all measurable) ----
        self.lr_schedule = LRSchedule(args.lr, args.lr_warmup,
                                      args.iterations, args.lr_min_ratio)
        self.entropy_schedule = EntropySchedule(
            args.entropy_coef, args.entropy_final, args.iterations)
        self.ref_agent = None
        if args.ema_decay > 0:
            self.ema = EMA([self.agent.policy, self.agent.value],
                           decay=args.ema_decay)
            self.ema_agent = PPOAgent(self.policy_dim, self.n_actions,
                                      hidden=args.hidden, cfg=self.ppo_cfg,
                                      seed=0)
        else:
            self.ema = None
            self.ema_agent = None
        self.rft = (RejectionBuffer(self.policy_dim, max_steps=args.rft_buffer)
                    if args.rft_frac > 0 else None)
        self.rft_loss = None

        # ---- weight-space evolution (PBT-lite) ----
        # Built after BC / checkpoint load: a population cloned from an
        # untrained net is noise, not evolution.
        self.pbt = None
        self.pbt_stats = None

        # ---- self-evolution: the agent's own opponent pool ----
        self.pool = None
        self._snap_seq = 0
        self._evo_seq = 0
        self._loaded_pool_state = None
        if args.selfevolve:
            self.pool = SelfPlayPool(beta=args.pfsp_beta,
                                     max_size=args.pool_size,
                                     competence_floor=args.pfsp_floor)
            self.pool.add("random",
                          lambda: RandomOpponent(n_actions=self.n_actions),
                          origin="random")
            # A frozen copy of the untrained policy is the seed opponent.
            # Self-play can start from it without any authored curriculum.
            seed = self._frozen_self()
            self.pool.add("self0", lambda s=seed: SnapshotOpponent(s, encode=self.encoder),
                          origin="self")

        if args.resume:
            self._load_state()
        elif args.bc_episodes > 0:
            # Bootstrap the neural policy from a cheap scripted expert.  PPO
            # then fine-tunes it, instead of asking a randomly initialised
            # network to discover "walk close, then kick" from sparse rewards.
            self.bc_loss = self._behavior_clone()
        elif args.latent_policy:
            # Zero-data run still needs a coordinate system before it can act.
            # Random play is enough: an encoder only has to be *informative*,
            # not expert.
            obs, acts, nxt, rew = self._random_transitions(
                max(2, args.imagine_batch // 16))
            self._fit_encoder(obs, acts, nxt, rew)
            self.bc_loss = None

        # The RLHF anchor is the behaviour-cloned policy, frozen before any
        # PPO update.  It has to be created after BC / checkpoint loading.
        if args.kl_coef > 0:
            self.ref_agent = PPOAgent(self.policy_dim, self.n_actions,
                                      hidden=args.hidden, cfg=self.ppo_cfg,
                                      seed=0)
            self.ref_agent.policy.copy_from(self.agent.policy)
        if self.ema is not None and self.ema.updates == 0:
            self.ema.update()

        if self.args.pbt:
            self.pbt = PBTPopulation(self.agent, size=self.args.pbt_size,
                                     sigma=self.args.pbt_sigma,
                                     seed=self.args.seed + 7)

        if self.pool is not None:
            # Rebuild the frozen-self entries that came back with the league.
            self._snap_seq = len(self.league)
            for i, snap in enumerate(self.league):
                name = "league%d" % i
                if self.pool.get(name) is None:
                    self.pool.add(name, lambda s=snap: SnapshotOpponent(s, encode=self.encoder),
                                  origin="self")
            self.pool.load_stats(self._loaded_pool_state)

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
        if self.ema is not None:
            arrays.update(self.ema.state_dict("ema"))
        if self.rnd is not None:
            arrays.update(self.rnd.state_dict("rnd"))
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
            "budget": self.ledger.state_dict(),
            "ema_updates": 0 if self.ema is None else self.ema.updates,
            "rft_loss": self.rft_loss,
            "rft_size": 0 if self.rft is None else len(self.rft),
            "pool": None if self.pool is None else self.pool.state_dict(),
            "snap_seq": self._snap_seq,
            "evo_seq": self._evo_seq,
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
        self._loaded_pool_state = state.get("pool")
        self._evo_seq = int(state.get("evo_seq", 0))
        cur_state = state.get("curriculum")
        if self.curriculum is not None and cur_state:
            self.curriculum.load_state_dict(cur_state)
        self.ledger.load_state_dict(state.get("budget"))
        if "rng_state" in state:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = state["rng_state"]
        self.agent.load_state_dict(arrays, "agent")
        if self.world_model is not None and "world__meta" in arrays:
            self.world_model.load_state_dict(arrays, "world")
        if self.ema is not None and "ema__updates" in arrays:
            self.ema.load_state_dict(arrays, "ema")
            self.ema.copy_to([self.ema_agent.policy, self.ema_agent.value])
        if self.rnd is not None and "rnd__meta" in arrays:
            self.rnd.load_state_dict(arrays, "rnd")
        self.rft_loss = state.get("rft_loss")

        self.league = []
        i = 0
        while ("league%d__meta" % i) in arrays:
            snap = PPOAgent(self.policy_dim, self.n_actions,
                            hidden=self.args.hidden, cfg=self.ppo_cfg, seed=0)
            snap.load_state_dict(arrays, "league%d" % i)
            self.league.append(snap)
            i += 1

        if (isinstance(self.world_model, LatentWorldModel)
                and self.args.latent_policy):
            self.encoder = self.world_model.encoder
            self.world_model.freeze_encoder = True
        if self.pbt is not None:
            self.pbt = PBTPopulation(self.agent, size=self.args.pbt_size,
                                     sigma=self.args.pbt_sigma,
                                     seed=self.args.seed + 7)

    # ------------------------------------------------------- imitation warm-up
    def _pi_obs(self, raw):
        """Observation as the policy sees it: raw arena state, or its latent."""
        raw = np.asarray(raw, dtype=np.float32)
        if self.encoder is None:
            return raw
        return self.encoder.forward(raw.reshape(1, -1))[0]

    def _fit_encoder(self, obs, acts, next_obs, rewards, epochs=None):
        """Self-supervised warm-up of the latent encoder, then freeze it.

        The policy is about to consume ``z`` instead of the 33 hand-crafted
        numbers.  If the encoder kept training afterwards, the policy's input
        coordinate system would drift under it every iteration.
        """
        epochs = self.args.encoder_epochs if epochs is None else epochs
        scaled = np.asarray(rewards, dtype=np.float64) * self.args.reward_scale
        out = {}
        for _ in range(max(1, epochs)):
            out = self.world_model.train(obs, acts, next_obs, self.rng,
                                         epochs=1, rewards=scaled)
        self.encoder = self.world_model.encoder
        self.world_model.freeze_encoder = True
        self.encoder_metrics = out
        return out

    def _random_transitions(self, episodes: int):
        """Fallback for zero-data runs: random play is enough to learn a
        coordinate system, it does not need to be good play."""
        obs, acts, next_obs, rews = [], [], [], []
        for _ in range(episodes):
            seed = int(self.rng.integers(2 ** 31))
            arena = self.arena_cls(seed=seed, max_frames=max(200, self.args.max_frames // 2),
                                   start_gap=self.args.train_gap,
                                   emotion=self.args.emotion,
                                   physics=self.args.physics,
                                   arena3d=self.args.arena3d,
                                   env_cfg=self.env_cfg)
            o0, o1 = arena.reset(seed)
            while True:
                a0 = int(self.rng.integers(self.n_actions))
                a1 = int(self.rng.integers(self.n_actions))
                n0, n1, r0, r1, done, _info = arena.step(a0, a1)
                obs.append(o0); acts.append(a0); next_obs.append(n0); rews.append(r0)
                obs.append(o1); acts.append(a1); next_obs.append(n1); rews.append(r1)
                o0, o1 = n0, n1
                if done:
                    break
        return (np.stack(obs).astype(np.float32),
                np.asarray(acts, dtype=np.int64),
                np.stack(next_obs).astype(np.float32),
                np.asarray(rews, dtype=np.float64))

    def _behavior_clone(self):
        gap = self.args.train_gap
        pocket = self.args.env == "articulated"
        obs, acts, next_obs, rews = [], [], [], []
        demo_rng = np.random.default_rng(self.args.seed + 100003)
        # Vary who the expert fights.  Cloning a mirror match teaches the net
        # to read the opponent's pose instead of the distance, which then
        # collapses against a different scripted style.
        pool = [
            RandomOpponent(n_actions=self.n_actions),
            ScriptedOpponent(SCRIPTED_GENES[0], reach_scale=self.reach_scale),
            ScriptedOpponent(SCRIPTED_GENES[1], reach_scale=self.reach_scale),
        ]
        for i in range(self.args.bc_episodes):
            seed = int(demo_rng.integers(2 ** 31))
            kind = i % (len(pool) + 1)
            for side in (0, 1):
                arena = self.arena_cls(seed=seed, max_frames=self.args.max_frames,
                                       start_gap=gap, emotion=self.args.emotion,
                                       physics=self.args.physics,
                                       arena3d=self.args.arena3d,
                                       env_cfg=self.env_cfg)
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
                    else:
                        a0 = (opp.act(arena.env.p0, arena.env.p1, o0) if opp
                              else int(heuristic_action(o0, pocket=pocket)))
                        a1 = int(heuristic_action(o1, pocket=pocket))
                    me = o0 if side == 0 else o1
                    me_act = a0 if side == 0 else a1
                    n0, n1, r0, r1, done, _info = arena.step(a0, a1)
                    nm = n0 if side == 0 else n1
                    obs.append(me)
                    acts.append(me_act)
                    next_obs.append(nm)
                    rews.append(r0 if side == 0 else r1)
                    o0, o1 = n0, n1
                    if done:
                        break
        if self.args.latent_policy:
            if not obs:
                obs, acts, next_obs, rews = self._random_transitions(4)
            self._fit_encoder(np.stack(obs).astype(np.float32),
                              np.asarray(acts, dtype=np.int64),
                              np.stack(next_obs).astype(np.float32), rews)
        if not obs:
            return None
        x = np.stack(obs).astype(np.float32)
        y = np.asarray(acts, dtype=np.int64)
        if self.args.mirror_bc:
            # Mirror augmentation: the arena is left/right symmetric and the
            # actions are opponent-relative, so the reflected state is a
            # second, equally valid demonstration at zero environment cost.
            x = np.concatenate([x, mirror_batch(x, self.obs_dim)])
            y = np.concatenate([y, y])
        if self.n_actions > N_ACTIONS:
            # The scripted expert only knows the original six buttons.  Without
            # this, JUMP / DUCK / low kicks would start with logits near -inf
            # and PPO would need hundreds of episodes to try them at all.
            # Random labels on 10% of states keep them alive without erasing
            # the walk-in-and-punch prior.
            noise = self.rng.random(len(y)) < 0.10
            y[noise] = self.rng.integers(0, self.n_actions, int(noise.sum()))
        if self.encoder is not None:
            x = self.encoder.forward(x)
        onehot_cache = np.eye(self.agent.n_actions, dtype=np.float64)
        epoch_loss = 0.0
        for _epoch in range(self.args.bc_epochs):
            epoch_loss = self._fit_policy(x, y, onehot_cache,
                                          lr=self.args.bc_lr)
        return round(epoch_loss, 4)

    def _fit_policy(self, x, y, onehot_cache=None, lr=1e-3):
        """One epoch of supervised updates on (state, action) pairs."""
        if onehot_cache is None:
            onehot_cache = np.eye(self.agent.n_actions, dtype=np.float64)
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
            self.agent.policy_adam.step(grads, lr=lr,
                                        max_norm=self.args.max_grad_norm)
        return float(np.mean(batch_losses)) if batch_losses else 0.0

    # ------------------------------------------------------------- opponents
    def _scripted_pool(self):
        if self.curriculum is not None:
            return self.curriculum.scripted_pool()
        return SCRIPTED_GENES

    def _holdout_gene(self):
        if self.args.holdout_level >= 0:
            # A fixed, harder rung for ablations: the curriculum would move the
            # goalposts whenever the policy improves, which makes A/B useless.
            return CURRICULUM_LEVELS[self.args.holdout_level]["gene"]
        if self.curriculum is not None:
            return self.curriculum.holdout_gene()
        return SCRIPTED_GENES[1]

    def _sample_opponent(self, iteration: int, k: int):
        if self.pool is not None:
            if iteration < self.args.warmup:
                return "random", RandomOpponent(n_actions=self.n_actions)
            entry = self.pool.sample(self.rng, mode=self.args.pfsp_mode)
            return entry.name, entry.make()
        if iteration < self.args.warmup:
            return "random", RandomOpponent(n_actions=self.n_actions)
        r = float(self.rng.random())
        if self.league and r < 0.45:
            idx = int(self.rng.integers(len(self.league)))
            return "league", SnapshotOpponent(self.league[idx], encode=self.encoder)
        if r < 0.80:
            pool = self._scripted_pool()
            gene = pool[int(self.rng.integers(len(pool)))]
            return "scripted", ScriptedOpponent(gene, reach_scale=self.reach_scale)
        return "random", RandomOpponent(n_actions=self.n_actions)

    # --------------------------------------------------------------- episodes
    def _surprise(self, obs, action, next_obs) -> float:
        """Curiosity bonus, normalised by its own running mean.

        Two sources, and the difference is the whole point:

        * ``wm`` -- latent world-model prediction error.  Rewards transitions
          the model finds *discontinuous*; a substitution (teleport) maximises
          it, so ``wm`` curiosity teaches the agent to teleport, not to win.
        * ``rnd`` -- predictor error against a frozen random target network.
          Rewards *unfamiliar states*.  Being one frame later does not count as
          novel, so teleportation gets no free bonus.

        In both cases only the above-average part (``ratio - 1``) is paid, so a
        constant offset cannot be farmed by standing still.
        """
        if self.args.surprise_coef <= 0:
            return 0.0
        mode = getattr(self.args, "surprise_mode", "wm")
        if mode == "rnd" and self.rnd is not None:
            err = self.rnd.error(obs)
            self.rnd.update(obs)
            self._rnd_ema = 0.995 * self._rnd_ema + 0.005 * err
            ratio = err / (self._rnd_ema + 1e-6)
        elif mode == "wm" and isinstance(self.world_model, LatentWorldModel):
            err = float(self.world_model.error(obs, action, next_obs))
            self._surprise_ema = 0.995 * self._surprise_ema + 0.005 * err
            ratio = err / (self._surprise_ema + 1e-6)
        else:
            return 0.0
        bonus = self.args.surprise_coef * min(max(ratio - 1.0, 0.0), 2.0)
        self._surprise_sum += bonus
        return bonus

    def _play(self, opponent, seed: int, collect: bool):
        arena = self.arena_cls(seed=seed, max_frames=self.args.max_frames,
                            start_gap=self.args.train_gap,
                            emotion=self.args.emotion,
                            physics=self.args.physics,
                            arena3d=self.args.arena3d,
                            env_cfg=self.env_cfg)
        o0, o1 = arena.reset(seed)
        side = int(self.rng.integers(2))
        ep = Episode()
        total_reward = 0.0
        info = {"winner": -1}
        while True:
            me_raw = o0 if side == 0 else o1
            me_obs = self._pi_obs(me_raw)
            action, logp, value = self.agent.act(me_obs, self.rng)

            if side == 0:
                opp_action = opponent.act(arena.env.p1, arena.env.p0, o1)
                a0, a1 = action, opp_action
            else:
                opp_action = opponent.act(arena.env.p0, arena.env.p1, o0)
                a0, a1 = opp_action, action

            n0, n1, r0, r1, done, info = arena.step(a0, a1)
            reward = r0 if side == 0 else r1
            next_raw = n0 if side == 0 else n1
            surprise = self._surprise(me_raw, action, next_raw)
            reward += surprise
            total_reward += reward
            if collect:
                ref_logp = (self.ref_agent.logp_of(me_obs, action)
                            if self.ref_agent is not None else 0.0)
                ep.steps.append(Step(
                    obs=np.asarray(me_obs, dtype=np.float32).copy(),
                    action=action, logp=logp, value=value,
                    reward=reward * self.args.reward_scale, done=done,
                    ref_logp=ref_logp,
                    raw_obs=np.asarray(me_raw, dtype=np.float32).copy(),
                ))
            o0, o1 = n0, n1
            if done:
                break

        ep.last_obs = self._pi_obs(o0 if side == 0 else o1)
        winner = int(info.get("winner", -1))
        if winner < 0:
            outcome = 0.5
        elif winner == side:
            outcome = 1.0
        else:
            outcome = 0.0
        hp = info.get("hp")
        if hp is not None and len(hp) == 2:
            # A continuous signal next to the binary win: how much health was
            # left.  Near-identical policies can be ranked by this even when
            # the win rate is saturated at 0 or 1.
            margin = (float(hp[side]) - float(hp[1 - side])) / 100.0
        else:
            margin = 1.0 if outcome == 1.0 else (-1.0 if outcome == 0.0 else 0.0)
        me_fighter = arena.env.p0 if side == 0 else arena.env.p1
        affect = (me_fighter.affect.vector().copy()
                  if getattr(me_fighter, "affect", None) is not None else None)
        return ep, {
            "outcome": outcome,
            "margin": margin,
            "frames": arena.frames,
            "return": total_reward,
            "winner": winner,
            "affect": affect,
            "guard_breaks": int(info.get("guard_breaks", 0)),
            "counters": int(info.get("counters", 0)),
            "wall_bounces": int(info.get("wall_bounces", 0)),
            "wall_slams": int(info.get("wall_slams", 0)),
            "sidesteps": int(info.get("sidesteps", 0)),
            "clashes": int(info.get("clashes", 0)),
            "head_hits": int(info.get("head_hits", 0)),
            "jumps": int(info.get("jumps", 0)),
            "low_hits": int(info.get("low_hits", 0)),
            "flying_hits": int(info.get("flying_hits", 0)),
            "subs": int(info.get("subs", 0)),
            "sub_avoided": int(info.get("sub_avoided", 0)),
        }

    def _collect(self, n: int, iteration: int):
        episodes = []
        outcomes = []
        margins = []
        frames = []
        returns = []
        clashes = []
        heads = []
        jumps = []
        lows = []
        flyings = []
        subs = []
        avoided = []
        affects = []
        breaks = []
        counters = []
        walls = []
        slams = []
        steps_z = []
        for k in range(n):
            name, opp = self._sample_opponent(iteration, k)
            seed = int(self.rng.integers(2 ** 31))
            opp.reset(int(self.rng.integers(2 ** 31)))
            ep, stat = self._play(opp, seed, collect=True)
            if self.pool is not None:
                self.pool.record(name, stat["outcome"])
            episodes.append(ep)
            outcomes.append(stat["outcome"])
            margins.append(stat["margin"])
            frames.append(stat["frames"])
            returns.append(stat["return"])
            clashes.append(stat["clashes"])
            heads.append(stat["head_hits"])
            jumps.append(stat["jumps"])
            lows.append(stat["low_hits"])
            flyings.append(stat["flying_hits"])
            subs.append(stat["subs"])
            avoided.append(stat["sub_avoided"])
            breaks.append(stat["guard_breaks"])
            counters.append(stat["counters"])
            walls.append(stat["wall_bounces"])
            slams.append(stat["wall_slams"])
            steps_z.append(stat["sidesteps"])
            if stat["affect"] is not None:
                affects.append(stat["affect"])

        # What the policy actually presses.  This is the only direct evidence
        # that the new degrees of freedom are being used rather than ignored.
        counts = np.zeros(self.n_actions, dtype=np.int64)
        for ep in episodes:
            for step in ep.steps:
                counts[step.action] += 1
        if self.args.env == "freeform":
            names = (FREEFORM_ACTION_NAMES_3D if self.args.arena3d
                     else FREEFORM_ACTION_NAMES)
        else:
            names = ACTION_NAMES
        total = int(counts.sum()) or 1
        hist = {names[i]: round(float(counts[i]) / total, 3)
                for i in range(min(len(names), self.n_actions))}

        return episodes, {
            "win": float(np.mean(outcomes)) if outcomes else 0.0,
            "margin": float(np.mean(margins)) if margins else 0.0,
            "frames": float(np.mean(frames)) if frames else 0.0,
            "return": float(np.mean(returns)) if returns else 0.0,
            "clashes": float(np.mean(clashes)) if clashes else 0.0,
            "head_hits": float(np.mean(heads)) if heads else 0.0,
            "jumps": float(np.mean(jumps)) if jumps else 0.0,
            "low_hits": float(np.mean(lows)) if lows else 0.0,
            "flying_hits": float(np.mean(flyings)) if flyings else 0.0,
            "subs": float(np.mean(subs)) if subs else 0.0,
            "sub_avoided": float(np.mean(avoided)) if avoided else 0.0,
            "guard_breaks": float(np.mean(breaks)) if breaks else 0.0,
            "counters": float(np.mean(counters)) if counters else 0.0,
            "wall_bounces": float(np.mean(walls)) if walls else 0.0,
            "wall_slams": float(np.mean(slams)) if slams else 0.0,
            "sidesteps": float(np.mean(steps_z)) if steps_z else 0.0,
            "affect": (np.mean(affects, axis=0) if affects else None),
            "steps": int(sum(len(ep.steps) for ep in episodes)),
            "returns": returns,
            "act_hist": hist,
        }

    def _evaluate(self, n: int, kind: str, agent=None):
        agent = agent if agent is not None else self.agent
        # A fixed seed suite, independent of the training RNG, so the same
        # checkpoint is always scored on the same matches.
        outcomes = []
        margins = []
        frames = []
        clashes = []
        heads = []
        subs = []
        walls = []
        slams = []
        steps_z = []
        eval_rng = np.random.default_rng(0)
        for seed in range(2000, 2000 + n):
            for side in (0, 1):
                if kind == "random":
                    opp = RandomOpponent(n_actions=self.n_actions)
                else:
                    opp = ScriptedOpponent(self._holdout_gene(),
                                          reach_scale=self.reach_scale)
                opp.reset(seed * 2 + side)
                arena = self.arena_cls(seed=seed, max_frames=self.args.max_frames,
                                    start_gap=self.args.eval_gap,
                                    emotion=self.args.emotion,
                                    physics=self.args.physics,
                                    arena3d=self.args.arena3d,
                                    env_cfg=self.env_cfg)
                o0, o1 = arena.reset(seed)
                info = {"winner": -1}
                while True:
                    me_raw = o0 if side == 0 else o1
                    me_obs = self._pi_obs(me_raw)
                    action, _logp, _value = agent.act(me_obs, eval_rng, deterministic=True)
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
                hp = info.get("hp")
                if hp is not None and len(hp) == 2:
                    margins.append((float(hp[side]) - float(hp[1 - side]))
                                   / 100.0)
                else:
                    margins.append(1.0 if w == side else
                                   (-1.0 if w >= 0 else 0.0))
                frames.append(arena.frames)
                clashes.append(int(info.get("clashes", 0)))
                heads.append(int(info.get("head_hits", 0)))
                subs.append(int(info.get("subs", 0)))
                walls.append(int(info.get("wall_bounces", 0)))
                slams.append(int(info.get("wall_slams", 0)))
                steps_z.append(int(info.get("sidesteps", 0)))
        summary = summarise(outcomes, seed=0)
        margin_summary = summarise(margins, seed=0)
        return {
            "win": float(np.mean(outcomes)) if outcomes else 0.0,
            "win_ci_low": summary["ci_low"],
            "win_ci_high": summary["ci_high"],
            "margin": float(margin_summary["mean"]),
            "margin_ci_low": margin_summary["ci_low"],
            "margin_ci_high": margin_summary["ci_high"],
            "frames": float(np.mean(frames)) if frames else 0.0,
            "clashes": float(np.mean(clashes)) if clashes else 0.0,
            "head_hits": float(np.mean(heads)) if heads else 0.0,
            "subs": float(np.mean(subs)) if subs else 0.0,
            "wall_bounces": float(np.mean(walls)) if walls else 0.0,
            "wall_slams": float(np.mean(slams)) if slams else 0.0,
            "sidesteps": float(np.mean(steps_z)) if steps_z else 0.0,
            "games": len(outcomes),
            "outcomes": outcomes,
        }

    # --------------------------------------------------------------- training
    def _world_model_step(self, episodes):
        """Return the model's prediction error in its own space.

        The plain model reports observation MSE; the latent model reports
        latent MSE plus the reconstruction error, so the two numbers can be
        compared across 1-D / 2-D / 3-D arenas even though observation sizes
        differ.
        """
        if self.world_model is None:
            return None
        obs, actions, next_obs, rewards = [], [], [], []
        for ep in episodes:
            for i, step in enumerate(ep.steps):
                if i + 1 >= len(ep.steps):
                    continue          # no real successor: skip instead of
                                      # mixing a policy-space bootstrap state in
                nxt = ep.steps[i + 1]
                obs.append(step.raw_obs if step.raw_obs is not None else step.obs)
                actions.append(step.action)
                next_obs.append(nxt.raw_obs if nxt.raw_obs is not None else nxt.obs)
                rewards.append(step.reward / max(self.args.reward_scale, 1e-9))
        if not obs:
            return None
        extra = ({}
                 if not isinstance(self.world_model, LatentWorldModel)
                 else {"rewards": np.asarray(rewards, dtype=np.float64)})
        out = self.world_model.train(
            np.stack(obs), np.asarray(actions, dtype=np.int64), np.stack(next_obs),
            self.rng, epochs=self.args.world_epochs, **extra,
        )
        if isinstance(out, dict):
            self.recon_mse = (None if getattr(self.world_model,
                                              "freeze_encoder", False)
                              else float(out.get("recon_mse", 0.0)))
            return float(out.get("latent_mse", 0.0))
        self.recon_mse = None
        return out

    def _entropy(self, episodes):
        obs = []
        for ep in episodes:
            for step in ep.steps[:32]:
                obs.append(step.obs)
        if not obs:
            return 0.0
        return float(np.mean([self.agent.entropy_of(o) for o in obs[:128]]))

    def _frozen_self(self):
        snap = PPOAgent(self.policy_dim, self.n_actions, hidden=self.args.hidden,
                        cfg=self.ppo_cfg, seed=0)
        snap.load_state_dict(self.agent.state_dict(), "agent")
        return snap

    def _snapshot(self):
        snap = self._frozen_self()
        self.league.append(snap)
        if len(self.league) > self.args.league_size:
            self.league.pop(0)
        if self.pool is not None:
            name = "league%d" % self._snap_seq
            self._snap_seq += 1
            entry = self.pool.add(name, lambda s=snap: SnapshotOpponent(s, encode=self.encoder),
                                  origin="self")
            self._cap_pool_entry(entry)

    def _cap_pool_entry(self, entry):
        """Keep the pool bounded without ever evicting the random anchor."""
        if self.pool is None or len(self.pool) <= self.args.pool_size:
            return
        victim = next((e for e in self.pool.entries
                       if e.origin == "evolved"), None)
        if victim is None:
            victim = next((e for e in self.pool.entries
                           if e.origin == "self" and e is not entry), None)
        if victim is not None:
            self.pool.remove(victim.name)

    def _spawn_candidate(self):
        """Mutate the frontier's gene: the agent writes its own next lesson."""
        frontier = self.pool.frontier()
        if frontier is not None and frontier.gene is not None:
            gene = mutate_gene(frontier.gene, self.rng, self.args.evolve_sigma)
        else:
            gene = random_gene(self.rng)
        name = "evo%d" % self._evo_seq
        self._evo_seq += 1
        entry = self.pool.add(
            name,
            lambda g=gene: ScriptedOpponent(g, reach_scale=self.reach_scale),
            gene=gene, origin="evolved")
        self._cap_pool_entry(entry)
        return name

    # --------------------------------------------------------------- imagine
    def _imagine(self, episodes):
        """Dyna-style rollout in latent space.

        Real transitions update the world model; the model then generates extra
        transitions the policy can learn from without touching the arena.  The
        reward comes from the model's reward head, so the numbers are cheap but
        *wrong* in a systematic way -- that is why they are weighted down and
        kept to a couple of steps.
        """
        steps = 0
        if (self.args.imagine_steps <= 0
                or not isinstance(self.world_model, LatentWorldModel)):
            return [], 0
        starts = [s for ep in episodes for s in ep.steps if len(ep.steps) > 2]
        if not starts:
            return [], 0
        n = min(len(starts), self.args.imagine_batch)
        picks = self.rng.integers(0, len(starts), size=n)
        ep = Episode()
        for k in picks:
            src = starts[int(k)]
            raw = src.raw_obs if src.raw_obs is not None else src.obs
            z = self.world_model.encode(raw)
            for t in range(self.args.imagine_steps):
                obs = z if self.encoder is not None else self.world_model.decode(z)
                action, logp, value = self.agent.act(obs, self.rng)
                z_next, r = self.world_model.step_latent(z, action)
                ep.steps.append(Step(
                    obs=np.asarray(obs, dtype=np.float32).copy(),
                    action=action, logp=logp, value=value,
                    reward=r * self.args.reward_scale * self.args.imagine_coef,
                    done=False, raw_obs=None,
                ))
                z = z_next
                steps += 1
        if steps:
            ep.last_obs = np.asarray(ep.steps[-1].obs, dtype=np.float32)
        return [ep], steps

    # ------------------------------------------------------------------- pbt
    def _pbt_step(self, iteration: int):
        """Score the population, keep winners, mutate survivors."""
        games = max(2, self.args.pbt_games)
        metric = self.args.fitness

        def fitness(m):
            return self._evaluate(games, "holdout", agent=m)[metric]

        champion, report = self.pbt.evolve(fitness)
        report["pbt_adopted"] = False
        if self.args.pbt_adopt:
            current = self._evaluate(games, "holdout")[metric]
            if report["pbt_best"] >= current + 0.02:
                self.agent.load_state_dict(champion.state_dict(), "agent")
                report["pbt_adopted"] = True
        if self.pool is not None:
            name = "pbt%d" % self._evo_seq
            self._evo_seq += 1
            snap = clone_agent(champion)
            entry = self.pool.add(name, lambda s=snap: SnapshotOpponent(s, encode=self.encoder),
                                  origin="evolved")
            self._cap_pool_entry(entry)
            report["pbt_opponent"] = name
        self.pbt_stats = report
        return report

    def train_iteration(self, iteration: int):
        self.ledger.start_iteration()
        self._surprise_sum = 0.0   # per-iteration bonus, not a running total
        with self.ledger.phase("collect"):
            episodes, stats = self._collect(self.args.episodes, iteration)
        real_episodes = episodes
        imagine_n = 0
        if self.args.imagine_steps > 0:
            with self.ledger.phase("imagine"):
                imagined, imagine_n = self._imagine(episodes)
                episodes = episodes + imagined
        with self.ledger.phase("ppo"):
            lr = self.lr_schedule.at(iteration)
            self.agent.policy_adam.lr = lr
            self.agent.value_adam.lr = lr
            self.ppo_cfg.entropy_coef = self.entropy_schedule.at(iteration)
            ppo = self.agent.train(episodes, self.rng)
        with self.ledger.phase("rft"):
            if self.rft is not None and stats.get("returns"):
                self.rft.add(episodes, stats["returns"], self.args.rft_frac)
                if len(self.rft) >= 32 and self.args.rft_epochs > 0:
                    for _ in range(self.args.rft_epochs):
                        x, y = self.rft.arrays()
                        self.rft_loss = round(self._fit_policy(
                            x, y, lr=self.args.rft_lr), 4)
        with self.ledger.phase("ema"):
            if self.ema is not None:
                self.ema.update()
                self.ema.copy_to([self.ema_agent.policy, self.ema_agent.value])
        with self.ledger.phase("world"):
            world_mse = self._world_model_step(real_episodes)
        if world_mse is not None:
            self.world_mse_history.append(float(world_mse))
            self.world_mse_history = self.world_mse_history[-50:]

        with self.ledger.phase("eval"):
            if (iteration % self.args.eval_every == 0
                    or self._last_eval is None):
                self._last_eval = (
                    self._evaluate(self.args.eval_episodes, "holdout"),
                    self._evaluate(self.args.eval_episodes, "random"),
                    (self._evaluate(self.args.eval_episodes, "holdout",
                                    agent=self.ema_agent)
                     if self.ema_agent is not None else None),
                )
            holdout, random_eval, ema_eval = self._last_eval
        entropy = self._entropy(real_episodes)

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
            with self.ledger.phase("snapshot"):
                self._snapshot()

        pool_stats = None
        if self.pool is not None:
            with self.ledger.phase("evolve"):
                if (iteration + 1) % self.args.evolve_every == 0:
                    self._spawn_candidate()
                self.pool.prune()
            pool_stats = self.pool.stats()

        pbt_stats = None
        if self.pbt is not None and (iteration + 1) % self.args.pbt_every == 0:
            with self.ledger.phase("pbt"):
                pbt_stats = self._pbt_step(iteration)

        rec = {
            "iter": iteration + 1,
            "episodes": self.args.episodes,
            "steps": stats["steps"],
            "train_win": round(float(stats["win"]), 4),
            "train_margin": round(float(stats["margin"]), 4),
            "train_return": round(float(stats["return"]), 2),
            "avg_frames": round(float(stats["frames"]), 1),
            "clashes": round(float(stats["clashes"]), 2),
            "eval_clashes": round(float(holdout["clashes"]), 2),
            "head_hits": round(float(stats["head_hits"]), 2),
            "eval_head_hits": round(float(holdout["head_hits"]), 2),
            "jumps": round(float(stats["jumps"]), 2),
            "low_hits": round(float(stats["low_hits"]), 2),
            "flying_hits": round(float(stats["flying_hits"]), 2),
            "act_hist": stats["act_hist"],
            "holdout_win": round(float(holdout["win"]), 4),
            "holdout_ci_low": round(float(holdout["win_ci_low"]), 4),
            "holdout_ci_high": round(float(holdout["win_ci_high"]), 4),
            "holdout_margin": round(float(holdout["margin"]), 4),
            "holdout_margin_ci_low": round(float(holdout["margin_ci_low"]), 4),
            "holdout_margin_ci_high": round(float(holdout["margin_ci_high"]), 4),
            "holdout_outcomes": holdout["outcomes"],
            "imagine_steps": int(imagine_n),
            "surprise": round(float(self._surprise_sum), 3),
            "policy_dim": int(self.policy_dim),
            **(pbt_stats or {}),
            "holdout_frames": round(float(holdout["frames"]), 1),
            "random_win": round(float(random_eval["win"]), 4),
            "entropy": round(float(entropy), 4),
            "entropy_coef": round(float(self.ppo_cfg.entropy_coef), 5),
            "lr": round(float(lr), 6),
            "ref_kl": round(float(ppo.get("ref_kl", 0.0)), 5),
            "ema_win": None if ema_eval is None else round(float(ema_eval["win"]), 4),
            "ema_updates": 0 if self.ema is None else self.ema.updates,
            "rft_loss": self.rft_loss,
            "world_mse": None if world_mse is None else round(float(world_mse), 5),
            "recon_mse": None if self.recon_mse is None else round(self.recon_mse, 5),
            "world_latent": (self.world_model.latent_dim
                             if isinstance(self.world_model, LatentWorldModel)
                             else None),
            "subs": round(float(stats.get("subs", 0.0)), 2),
            "eval_subs": round(float(holdout.get("subs", 0.0)), 2),
            "sub_avoided": round(float(stats.get("sub_avoided", 0.0)), 2),
            "guard_breaks": round(float(stats.get("guard_breaks", 0.0)), 2),
            "counters": round(float(stats.get("counters", 0.0)), 2),
            "wall_bounces": round(float(stats.get("wall_bounces", 0.0)), 2),
            "wall_slams": round(float(stats.get("wall_slams", 0.0)), 2),
            "eval_wall_slams": round(float(holdout.get("wall_slams", 0.0)), 2),
            "sidesteps": round(float(stats.get("sidesteps", 0.0)), 2),
            "eval_sidesteps": round(float(holdout.get("sidesteps", 0.0)), 2),
            "affect": (None if stats.get("affect") is None else {
                name: round(float(value), 3) for name, value
                in zip(AFFECT_KEYS, stats["affect"])}),
            "policy_loss": round(float(ppo["policy_loss"]), 5),
            "value_loss": round(float(ppo["value_loss"]), 5),
            "approx_kl": round(float(ppo["approx_kl"]), 5),
            "clip_frac": round(float(ppo["clip_frac"]), 4),
            "grad_norm": round(float(ppo["grad_norm"]), 4),
            "param_norm": round(float(self.agent.param_norm()), 4),
            "bc_loss": self.bc_loss,
            "league_size": len(self.league),
            **(pool_stats or {}),
            "curriculum_level": None if self.curriculum is None else self.curriculum.level,
            "curriculum_name": None if self.curriculum is None else self.curriculum.level_name,
            "curriculum_action": None if curriculum_step is None else curriculum_step["action"],
            "curriculum_reason": None if curriculum_step is None else curriculum_step["reason"],
            "health": health["status"],
            "health_reasons": health["reasons"],
            "eval_games": holdout["games"],
        }
        compute = self.ledger.finish_iteration(stats["steps"])
        compute["iter"] = iteration + 1
        compute["run_id"] = self.run_id
        self.last_compute = compute
        rec.update(self.ledger.skill(holdout["win"], holdout["games"]))
        return rec

    def run(self):
        for iteration in range(self.next_iter, self.args.iterations):
            t0 = time.time()
            rec = self.train_iteration(iteration)
            rec["elapsed_sec"] = round(time.time() - t0, 2)
            self.recorder.log(rec)
            self.recorder.log_compute(self.last_compute)
            self.next_iter = iteration + 1
            self._save_state()
            print(json.dumps(rec, ensure_ascii=False), flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description="Neural self-play in the duel arena")
    ap.add_argument("--env", choices=("duel", "articulated", "freeform"),
                    default="duel",
                    help="duel: 1-D reach check; articulated: limb-capsule "
                         "contact; freeform: articulated + jump/duck/substitution "
                         "(9 actions)")
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
    ap.add_argument("--latent-world", action="store_true",
                    help="compress observations through an 8-D bottleneck and "
                         "predict in latent space instead of raw observation space")
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--snapshot-every", type=int, default=2)
    ap.add_argument("--league-size", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="score every N iterations; >1 reuses the last score "
                         "and cuts the eval share (often the largest cost)")
    ap.add_argument("--health-window", type=int, default=10)
    ap.add_argument("--train-gap", type=float, default=2.2)
    ap.add_argument("--eval-gap", type=float, default=2.2)
    ap.add_argument("--bc-episodes", type=int, default=None,
                    help="scripted-expert episodes used for BC warm-up "
                         "(default 8; 0 disables demos entirely)")
    ap.add_argument("--bc-epochs", type=int, default=60)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--curriculum", action="store_true", default=True)
    ap.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    ap.add_argument("--curriculum-patience", type=int, default=2)
    ap.add_argument("--wm-level-threshold", type=float, default=0.02)

    # ---- LLM-style training stack ----
    ap.add_argument("--llm-stack", action="store_true",
                    help="enable the whole LLM recipe: AdamW, warmup+cosine LR, "
                         "KL anchor, EMA weights, rejection sampling (RFT), "
                         "mirror augmentation and entropy annealing")
    ap.add_argument("--weight-decay", type=float, default=None,
                    help="decoupled AdamW weight decay on weight matrices")
    ap.add_argument("--kl-coef", type=float, default=None,
                    help="RLHF anchor: coefficient on KL(policy || cloned prior)")
    ap.add_argument("--lr-warmup", type=int, default=None,
                    help="iterations of linear LR warmup")
    ap.add_argument("--lr-min-ratio", type=float, default=None,
                    help="cosine floor as a fraction of --lr (1.0 = flat)")
    ap.add_argument("--entropy-final", type=float, default=None,
                    help="anneal entropy_coef to this value over the run")
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="EMA decay for the shadow policy (0 = disabled)")
    ap.add_argument("--rft-frac", type=float, default=None,
                    help="keep this top fraction of each iteration's episodes "
                         "for rejection-sampling fine-tuning (0 = disabled)")
    ap.add_argument("--rft-epochs", type=int, default=None)
    ap.add_argument("--rft-lr", type=float, default=3e-4)
    ap.add_argument("--rft-buffer", type=int, default=20000)
    ap.add_argument("--selfevolve", action="store_true",
                    help="zero-data self-evolution: no BC demos, no authored "
                         "ladder; opponents come from self snapshots and "
                         "mutated genes, sampled by win-rate band")
    ap.add_argument("--pfsp-mode", choices=("pfsp", "progress", "uniform"),
                    default="pfsp",
                    help="how the self-play pool picks the next opponent")
    ap.add_argument("--pfsp-beta", type=float, default=2.0,
                    help="PFSP sharpening: weight ∝ (1 - win)^beta")
    ap.add_argument("--pfsp-floor", type=float, default=0.10,
                    help="opponents below this win rate are down-weighted; "
                         "0 = classic PFSP, which collapses from a cold start")
    ap.add_argument("--latent-policy", action="store_true",
                    help="policy consumes the learned 8-D latent instead of the "
                         "33 hand-crafted numbers (tests representation quality)")
    ap.add_argument("--encoder-epochs", type=int, default=8,
                    help="self-supervised warm-up epochs before freezing encoder")
    ap.add_argument("--surprise-coef", type=float, default=0.0,
                    help="curiosity bonus coefficient; 0 = off, 0.5 is a "
                         "reasonable value")
    ap.add_argument("--surprise-mode", choices=("wm", "rnd", "none"),
                    default="wm",
                    help="wm: latent prediction error (rewards teleports); "
                         "rnd: random-network distillation (rewards unfamiliar "
                         "states); none: disable")
    ap.add_argument("--emotion", action="store_true",
                    help="append four affect dims per fighter (composure, fear, "
                         "anger, momentum); freeform only, +8 obs dims")
    ap.add_argument("--physics", action="store_true",
                    help="freeform only: mass, Coulomb friction, wall "
                         "restitution, impulse knockback and hitstop")
    ap.add_argument("--arena3d", action="store_true",
                    help="freeform only: add the depth axis, a SIDESTEP "
                         "action and volumetric (z-aware) hit detection")
    ap.add_argument("--mass0", type=float, default=70.0,
                    help="freeform + --physics: weight class of seat 0 in kg")
    ap.add_argument("--mass1", type=float, default=70.0,
                    help="freeform + --physics: weight class of seat 1 in kg")
    ap.add_argument("--fitness", choices=("win", "margin"), default="win",
                    help="selection signal for PBT: binary win rate, or the "
                         "continuous health margin (hp difference / 100)")
    ap.add_argument("--pbt", action="store_true",
                    help="weight-space evolution: population, mutation, selection")
    ap.add_argument("--pbt-size", type=int, default=6)
    ap.add_argument("--pbt-sigma", type=float, default=0.03,
                    help="relative weight noise, as a fraction of tensor std")
    ap.add_argument("--pbt-every", type=int, default=4)
    ap.add_argument("--pbt-games", type=int, default=4,
                    help="games per member per generation (fixed seeds, paired)")
    ap.add_argument("--pbt-adopt", action="store_true",
                    help="replace the learner when the champion is clearly better")
    ap.add_argument("--imagine-steps", type=int, default=0,
                    help="latent rollout length per imagined start (0 = off)")
    ap.add_argument("--imagine-batch", type=int, default=48,
                    help="number of imagined starts per iteration")
    ap.add_argument("--imagine-coef", type=float, default=0.5,
                    help="down-weight for imagined rewards: the model is wrong")
    ap.add_argument("--pool-size", type=int, default=10)
    ap.add_argument("--evolve-every", type=int, default=1,
                    help="iterations between generated opponent candidates")
    ap.add_argument("--evolve-sigma", type=float, default=0.20,
                    help="gene mutation size, in fraction of each gene's span")
    ap.add_argument("--mirror-bc", action="store_true",
                    help="left/right mirror augmentation for the BC demos")
    ap.add_argument("--holdout-level", type=int, default=-1,
                    help="pin the holdout to CURRICULUM_LEVELS[i] instead of "
                         "letting the curriculum move it (for clean ablations)")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    trainer = NeuroTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
