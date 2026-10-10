"""EnvironmentLoop seeding."""

import gymnasium as gym
import numpy as np
import pytest
from conftest import make_vec_env

from mujax.loop import EnvironmentLoop


class _FirstObservation:
    """Stands in for the Actor; only records the observation of the first reset."""

    def observe_first(self, observation, info):
        self.observation = observation


def _first_observation(seed):
    actor = _FirstObservation()
    env = make_vec_env(2)
    EnvironmentLoop(env, actor, seed=seed)._start()
    env.close()
    return actor.observation


def test_seed_makes_first_reset_reproducible():
    np.testing.assert_array_equal(_first_observation(3), _first_observation(3))
    assert not np.array_equal(_first_observation(3), _first_observation(4))


def test_seed_is_used_only_on_the_first_reset():
    actor = _FirstObservation()
    env = make_vec_env(2)
    loop = EnvironmentLoop(env, actor, seed=3)
    first = loop._start()
    second = loop._start()
    env.close()
    assert not np.array_equal(first, second)


def test_single_env_is_wrapped_as_vector_env():
    actor = _FirstObservation()
    env = gym.make("CartPole-v1")
    loop = EnvironmentLoop(env, actor, seed=3)
    observation = loop._start()
    assert observation.shape == (1, 4)
    assert loop._num_envs == 1
    np.testing.assert_array_equal(observation, _first_observation(3)[:1])
    loop._env.close()


def test_next_step_vector_env_is_rejected():
    env = gym.make_vec("CartPole-v1", num_envs=1, vectorization_mode="sync")
    with pytest.raises(ValueError, match="SAME_STEP"):
        EnvironmentLoop(env, _FirstObservation())
    env.close()
