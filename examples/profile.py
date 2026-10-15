"""Profile a registered Gymnasium environment.

    python examples/profile.py
    python examples/profile.py smz --size S --batch-size 512 --num-actors 4

Custom environments use the same ExperimentConfig as training:

    from mujax import profile
    profile(experiment)
"""

from __future__ import annotations

import dataclasses
from typing import Annotated

import tyro

from mujax import ExperimentConfig, GMZConfig, SMZConfig, profile
from mujax.cli import make_vec_env

AgentConfig = tyro.conf.OmitSubcommandPrefixes[
    Annotated[GMZConfig, tyro.conf.subcommand("gmz", prefix_name=False)] | Annotated[SMZConfig, tyro.conf.subcommand("smz", prefix_name=False)]
]


@dataclasses.dataclass
class ProfileArgs:
    """Bottleneck profiler for a Gymnasium environment id."""

    agent: AgentConfig = dataclasses.field(default_factory=GMZConfig)
    env: str = "CartPole-v1"  # Gymnasium env id. Custom envs: call mujax.profile with your ExperimentConfig
    num_envs: int = 16  # Sub-envs per actor thread
    num_actors: int = 1
    num_steps: int | None = None  # Learner step budget for ETA calculation
    size: str = "S"  # Network size preset: XS, S, M, L, XL
    actor_iters: int = 100  # Timed actor steps after the first JIT compile
    learner_iters: int = 20  # Timed learner steps after the first JIT compile
    concurrent_seconds: float = 2.0  # Actors and learner together; 0 skips that part


def main() -> None:
    args = tyro.cli(ProfileArgs)
    config = args.agent.with_size(args.size)
    experiment = ExperimentConfig(
        config=config,
        environment_factory=lambda seed: make_vec_env(args.env, args.num_envs),
        max_num_learner_steps=args.num_steps,
        num_actors=args.num_actors,
        extra={"env_id": args.env},
    )
    profile(
        experiment,
        actor_iters=args.actor_iters,
        learner_iters=args.learner_iters,
        concurrent_seconds=args.concurrent_seconds,
    )


if __name__ == "__main__":
    main()
