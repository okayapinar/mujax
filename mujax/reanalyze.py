from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from mujax.algorithm import SearchPolicy
from mujax.config import MuZeroConfig
from mujax.replay import Buffer

_IDLE_SLEEP_SEC = 0.001  # wait while replay is not full enough to yield a sample


class Reanalyzer:
    """Batch iterator that samples from replay and re-searches a fraction of it with the current params.

    `next()` blocks until a sample is ready; raises StopIteration once `stop_event` is set.
    Runs in the learner thread; params live on the learner device.

    Args:
        policy: search policy built with the reanalyze simulation count.
        value_fn: (params, obs) -> search-free value; only for tail positions needed for bootstrapping.
        get_params: () -> (params, learner_steps).
    """

    def __init__(
        self,
        replay: Buffer,
        *,
        config: MuZeroConfig,
        policy: SearchPolicy,
        value_fn: Callable[[Any, Any], jnp.ndarray],
        get_params: Callable[[], tuple[Any, int]],
        device: jax.Device | None,
        random_key: jax.Array,
        stop_event: threading.Event,
    ):
        self._replay = replay
        self._policy = policy
        self._value_fn = value_fn
        self._get_params = get_params
        self._device = device
        self._key = random_key
        self._stop_event = stop_event
        self._num_reanalyze = min(config.batch_size, max(0, round(config.batch_size * config.reanalyze_ratio)))
        # The loss uses the MCTS policy only in 0..num_unroll; the remaining positions are needed only
        # for the n-step bootstrap value, and the network value suffices for them instead of search.
        self._num_search = min(config.num_unroll_steps + 1, config.sequence_length)
        self._refresh = jax.jit(self._refresh_trajectories)

    def __iter__(self) -> Reanalyzer:
        return self

    def __next__(self) -> Any:
        while not self._stop_event.is_set():
            self._key, sample_key, search_key = jax.random.split(self._key, 3)
            sample = self._replay.sample(sample_key)
            if sample is None:
                time.sleep(_IDLE_SLEEP_SEC)
                continue
            sample = jax.device_put(sample, self._device)
            if self._num_reanalyze > 0:
                params, learner_steps = self._get_params()
                sample = dataclasses.replace(sample, experience=self._refresh(params, sample.experience, search_key, learner_steps))
            return sample
        raise StopIteration

    def _refresh_trajectories(self, params: Any, traj: Any, key: jax.Array, learner_steps: Any) -> Any:
        n, k = self._num_reanalyze, self._num_search
        obs = traj.observation[:n]
        t = obs.shape[1]

        search = self._policy(params, obs[:, :k].reshape(n * k, -1), key, traj.invalid_actions[:n, :k].reshape(n * k, -1), learner_steps)
        value = search.value.reshape(n, k)
        if t > k:
            tail_value = self._value_fn(params, obs[:, k:].reshape(n * (t - k), -1)).reshape(n, t - k)
            value = jnp.concatenate([value, tail_value], axis=1)

        return traj._replace(
            policy_probs=traj.policy_probs.at[:n, :k].set(search.policy_probs.reshape(n, k, -1)),
            value=traj.value.at[:n].set(value),
        )
