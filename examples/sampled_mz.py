"""Train Sampled MuZero on a discrete-action environment and keep the best checkpoint.

python examples/sampled_mz.py                   # Acrobot-v1 (3 actions)
python examples/sampled_mz.py LunarLander-v3    # 4 actions; needs `pip install "gymnasium[box2d]"`

Sampled MuZero searches only over K actions sampled from the policy at every node (`num_sampled_actions`).
K is kept below the number of actions here so the sampling actually restricts the search; with K well above the
action count the search sees every action and the algorithm reduces to MuZero with a noisy prior.
"""

from __future__ import annotations

import sys

import gymnasium as gym

from mujax import CheckpointingConfig, ExperimentConfig, SampledMZConfig, run_experiment

NUM_STEPS = 20_000


def make_env(env_id: str, num_envs: int) -> gym.vector.VectorEnv:
    # SAME_STEP autoreset is required: the final observation of a truncated episode is read from `info["final_obs"]`.
    return gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )


def main(env_id: str = "Acrobot-v1") -> None:
    config = SampledMZConfig(
        batch_size=256,
        num_sampled_actions=2,  # K
        sample_temperature=1.0,  # beta = pi, as in the paper's experiments
    )
    experiment = ExperimentConfig(
        config=config.with_num_steps(NUM_STEPS),
        environment_factory=lambda seed: make_env(env_id, 16),
        eval_environment_factory=lambda seed: make_env(env_id, 1),
        max_num_learner_steps=NUM_STEPS,
        checkpointing=CheckpointingConfig(directory=f"checkpoints/sampled_mz_{env_id}", upload_to_wandb=False),
        checkpoint_extra={"env_id": env_id},
    )
    run_experiment(experiment)


if __name__ == "__main__":
    main(*sys.argv[1:2])
