from __future__ import annotations

import dataclasses
import os
from datetime import datetime
from typing import Annotated

import gymnasium as gym
import tyro

from mujax.algorithms import GMZConfig, MZConfig, SampledMZConfig, SMZConfig, algorithm_for
from mujax.checkpoint import CheckpointingConfig
from mujax.experiment import ExperimentConfig, run_experiment
from mujax.loggers import WandbLoggerFactory

AgentConfig = tyro.conf.OmitSubcommandPrefixes[
    Annotated[MZConfig, tyro.conf.subcommand("mz", prefix_name=False)]
    | Annotated[SMZConfig, tyro.conf.subcommand("smz", prefix_name=False)]
    | Annotated[GMZConfig, tyro.conf.subcommand("gmz", prefix_name=False)]
    | Annotated[SampledMZConfig, tyro.conf.subcommand("sampled_mz", prefix_name=False)]
]


@dataclasses.dataclass
class Args:
    """MuZero / Stochastic MuZero / Gumbel MuZero / Sampled MuZero training on Gymnasium environments.

    Example: `mujax --env CartPole-v1 gmz --num-simulations 32`. lr_decay_steps and
    temperature_decay_steps are scaled relative to --num-steps.
    """

    agent: AgentConfig
    env: str = "CartPole-v1"  # Gymnasium env id
    num_envs: int = 16
    num_steps: int = 100_000  # Total learner steps
    seed: int = 0
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "mujax"
    wandb_api_key: str | None = None  # If unset, wandb's own credentials (`wandb login`) are used
    wandb: bool = False  # Log to W&B; otherwise only the progress bar is shown


def make_vec_env(env_id: str, num_envs: int) -> gym.vector.VectorEnv:
    return gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="sync", vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})


def main(argv: list[str] | None = None) -> None:
    args = tyro.cli(Args, args=argv)

    algo = algorithm_for(args.agent).name
    config = args.agent.with_num_steps(args.num_steps)

    run_name = f"{algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger_factory = None
    if args.wandb:
        logger_factory = WandbLoggerFactory(
            project=args.wandb_project,
            name=run_name,
            config={"algo": algo, "env": args.env, "num_envs": args.num_envs, "seed": args.seed, **config.to_dict()},
            api_key=args.wandb_api_key,
        )

    experiment = ExperimentConfig(
        config=config,
        environment_factory=lambda seed: make_vec_env(args.env, args.num_envs),
        eval_environment_factory=lambda seed: make_vec_env(args.env, 1),
        max_num_learner_steps=args.num_steps,
        seed=args.seed,
        logger_factory=logger_factory,
        checkpointing=CheckpointingConfig(directory=os.path.join(args.checkpoint_dir, run_name)),
        checkpoint_extra={"env_id": args.env},
    )
    run_experiment(experiment)
    print("Tamamlandi")
