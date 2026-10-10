from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np

from mujax.actor import Actor
from mujax.loggers import Logger
from mujax.observers import EnvLoopObserver
from mujax.types import TimeStep, as_vector_env

EpisodeCallback = Callable[[dict[str, float]], None]


class Counter:
    """Thread-safe counter; actor, evaluator and learner share the same instance (`actor_steps`, `learner_steps`, ...)."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def increment(self, **counts: int) -> dict[str, int]:
        with self._lock:
            for key, value in counts.items():
                self._counts[key] = self._counts.get(key, 0) + int(value)
            return dict(self._counts)

    def get(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


class EnvironmentLoop:
    """Runs the env with the actor; reports each sub-env's episode result via counter, observers and logger.

    Accepts a single `gym.Env` (wrapped as a `num_envs=1` vector env, see `as_vector_env`) or a
    `gym.vector.VectorEnv` built with SAME_STEP autoreset:
    `gym.make_vec(env_id, num_envs, vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})`.

    Args:
        name: prefix of the counter keys (`<name>_steps`, `<name>_episodes`).
        episode_callback: Each episode result (counts included) is passed to this before being written to the logger;
            it may add new keys to the result.
        seed: passed to the first `env.reset`; sub-env i gets `seed + i`, later episodes continue from those RNGs.
            None: the env seeds itself (not reproducible).
    """

    def __init__(
        self,
        environment: gym.Env | gym.vector.VectorEnv,
        actor: Actor,
        *,
        name: str = "actor",
        counter: Counter | None = None,
        logger: Logger | None = None,
        observers: Sequence[EnvLoopObserver] = (),
        episode_callback: EpisodeCallback | None = None,
        stop_event: threading.Event | None = None,
        seed: int | None = None,
    ) -> None:
        environment = as_vector_env(environment)
        self._env = environment
        self._actor = actor
        self._name = name
        self._counter = counter or Counter()
        self._logger = logger
        self._observers = list(observers)
        self._episode_callback = episode_callback
        self._stop_event = stop_event or threading.Event()
        self._num_envs = int(environment.num_envs)
        self._seed = seed

    def run(self) -> None:
        """Runs until `stop_event` is set."""
        observation = self._start()
        while not self._stop_event.is_set():
            observation = self._step(observation)

    def _start(self) -> np.ndarray:
        observation, info = self._env.reset(seed=self._seed)
        self._seed = None
        self._actor.observe_first(observation, info)
        for observer in self._observers:
            observer.observe_first(self._env, self._num_envs)
        self._returns = np.zeros(self._num_envs, dtype=np.float64)
        self._lengths = np.zeros(self._num_envs, dtype=np.int64)
        self._start_times = np.full(self._num_envs, time.time())
        return observation

    def _step(self, observation: np.ndarray) -> np.ndarray:
        action = self._actor.select_action(observation)
        # SAME_STEP: next_observation of finished envs is already the first observation of the new episode.
        next_observation, reward, terminated, truncated, info = self._env.step(action)
        step = TimeStep(observation, action, reward, terminated, truncated, info, self._actor.last_search())
        self._actor.observe(step)
        for observer in self._observers:
            observer.observe(self._env, step)

        self._returns += np.asarray(reward, dtype=np.float64)
        self._lengths += 1
        done = np.asarray(terminated) | np.asarray(truncated)
        counts = self._counter.increment(**{f"{self._name}_steps": self._num_envs, f"{self._name}_episodes": int(done.sum())})
        for index in np.flatnonzero(done):
            self._finish_episode(int(index), counts)
        return next_observation

    def _finish_episode(self, index: int, counts: Mapping[str, int]) -> None:
        now = time.time()
        result: dict[str, float] = {
            "episode_return": float(self._returns[index]),
            "episode_length": float(self._lengths[index]),
            "steps_per_second": float(self._lengths[index] / max(1e-6, now - self._start_times[index])),
        }
        for observer in self._observers:
            result.update(observer.get_metrics(index))
        result.update(counts)
        if self._episode_callback is not None:
            self._episode_callback(result)
        if self._logger is not None:
            self._logger.write(result)
        self._returns[index] = 0.0
        self._lengths[index] = 0
        self._start_times[index] = now
