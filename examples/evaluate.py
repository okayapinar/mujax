"""Play episodes with a trained checkpoint.

python examples/evaluate.py checkpoints/cartpole
python examples/evaluate.py entity/project/run-checkpoint:best   # W&B artifact
"""

from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np

from mujax import as_vector_env, load_actor


def main(reference: str, num_episodes: int = 5) -> None:
    actor, metadata = load_actor(reference)
    env = as_vector_env(gym.make(metadata["extra"]["env_id"]))  # actor works on (num_envs, ...) batches
    for episode in range(num_episodes):
        observation, info = env.reset(seed=episode)
        actor.observe_first(observation, info)
        episode_return, done = 0.0, False
        while not done:
            observation, reward, terminated, truncated, info = env.step(
                actor.select_action(observation)
            )
            episode_return += float(reward[0])
            done = bool(np.logical_or(terminated, truncated)[0])
        print(f"episode {episode}: return={episode_return:.1f}")
    env.close()


if __name__ == "__main__":
    main(sys.argv[1])
