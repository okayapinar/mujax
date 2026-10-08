"""EnvironmentLoop seeding."""

import numpy as np
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
