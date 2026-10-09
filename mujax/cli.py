from __future__ import annotations

import dataclasses
import os
from datetime import datetime
from typing import Annotated

import gymnasium as gym
import tyro

from mujax.algorithms import GMZConfig, SMZConfig, algorithm_for
from mujax.checkpoint import CheckpointingConfig
from mujax.experiment import ExperimentConfig, run_experiment
from mujax.loggers import create_writer

AgentConfig = tyro.conf.OmitSubcommandPrefixes[
    Annotated[GMZConfig, tyro.conf.subcommand("gmz", prefix_name=False)]
    | Annotated[SMZConfig, tyro.conf.subcommand("smz", prefix_name=False)]
]


@dataclasses.dataclass
class Args:
    """Gumbel MuZero / Stochastic MuZero training on Gymnasium environments.

    Example: `mujax --env CartPole-v1 gmz --num-simulations 32`. lr_warmup_steps and
    temperature_decay_steps are scaled relative to the number of learner steps.
    """

    agent: AgentConfig
    env: str = "CartPole-v1"  # Gymnasium env id
    num_envs: int = 16  # Sub-envs per actor thread
    num_actors: int = 1  # Actor threads; the CPU search scales with the number of cores, actor steps are summed
    num_steps: int = 100_000  # Total learner steps
    # Total env steps (summed over envs); converted to learner steps with replay_ratio and overrides --num-steps.
    num_env_steps: int | None = None
    size: str = "M"  # Network size preset: S, M, L, XL (S is enough for classic-control envs and searches much faster)
    seed: int = 0
    checkpoint_dir: str = "checkpoints"
    console: bool = True  # Print metrics to the terminal
    wandb: bool = False  # Also log to W&B
    wandb_project: str = "mujax"


def make_vec_env(env_id: str, num_envs: int) -> gym.vector.VectorEnv:
    return gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="sync", vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})


def main(argv: list[str] | None = None) -> None:
    args = tyro.cli(Args, args=argv)

    algo = algorithm_for(args.agent).name
    config = args.agent.with_size(args.size)
    num_steps = args.num_steps if args.num_env_steps is None else config.num_learner_steps(args.num_env_steps)
    config = config.with_num_steps(num_steps)

    run_name = f"{algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = create_writer(console=args.console, wandb_project=args.wandb_project if args.wandb else None, wandb_name=run_name)

    experiment = ExperimentConfig(
        config=config,
        environment_factory=lambda seed: make_vec_env(args.env, args.num_envs),
        eval_environment_factory=lambda seed: make_vec_env(args.env, 1),
        max_num_learner_steps=num_steps,
        num_actors=args.num_actors,
        seed=args.seed,
        writer=writer,
        checkpointing=CheckpointingConfig(directory=os.path.join(args.checkpoint_dir, run_name)),
        extra={"env_id": args.env, "num_envs": args.num_envs, "num_actors": args.num_actors, "size": args.size},
    )
    run_experiment(experiment)
    print("Tamamlandi")
