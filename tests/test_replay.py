import numpy as np
import pytest
from conftest import make_vec_env

from mujax.replay import Buffer
from mujax.types import EnvironmentSpec, Transition, make_environment_spec

SPEC = EnvironmentSpec(obs_dim=3, num_actions=2)


def transition(num_envs: int) -> Transition:
    zeros = np.zeros(num_envs, np.float32)
    return Transition(
        observation=np.zeros((num_envs, SPEC.obs_dim), np.float32),
        action=np.zeros(num_envs, np.int32),
        reward=zeros,
        discount=zeros + 1,
        policy_probs=np.full((num_envs, SPEC.num_actions), 0.5, np.float32),
        value=zeros,
        invalid_actions=np.zeros((num_envs, SPEC.num_actions), np.float32),
        truncated=zeros,
        bootstrap_value=zeros,
    )


def test_sample_after_one_chunk():
    import jax

    buffer = Buffer(SPEC, num_envs=2, max_size=64, sample_batch_size=4, sequence_length=5)
    key = jax.random.PRNGKey(0)
    for _ in range(4):
        buffer.add(transition(2))
    assert buffer.sample(key) is None and buffer.size == 0

    buffer.add(transition(2))
    assert buffer.size == 10
    batch = buffer.sample(key).experience
    assert batch.observation.shape == (4, 5, SPEC.obs_dim)
    assert batch.policy_probs.shape == (4, 5, SPEC.num_actions)


def test_environment_spec():
    env = make_vec_env()
    assert make_environment_spec(env) == EnvironmentSpec(obs_dim=4, num_actions=2)
    env.close()


def test_environment_spec_rejects_continuous_actions():
    import gymnasium as gym

    env = gym.make_vec("Pendulum-v1", num_envs=1, vectorization_mode="sync")
    with pytest.raises(ValueError):
        make_environment_spec(env)
    env.close()
