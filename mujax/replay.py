"""Flashbax-based, thread-safe trajectory replay."""

from __future__ import annotations

import threading

import flashbax
import jax
import jax.numpy as jnp
import numpy as np
import psutil

from mujax.types import EnvironmentSpec, Transition

MAX_AUTO_SIZE = 5_000_000


def _item_template(spec: EnvironmentSpec) -> Transition:
    """Shapes and dtypes of a single time step."""
    return Transition(
        observation=jnp.zeros((spec.obs_dim,), jnp.float32),
        action=jnp.zeros((), jnp.int32),
        reward=jnp.zeros((), jnp.float32),
        discount=jnp.zeros((), jnp.float32),
        policy_probs=jnp.zeros((spec.num_actions,), jnp.float32),
        value=jnp.zeros((), jnp.float32),
        invalid_actions=jnp.zeros((spec.num_actions,), jnp.float32),
        truncated=jnp.zeros((), jnp.float32),
        bootstrap_value=jnp.zeros((), jnp.float32),
    )


def item_bytes(spec: EnvironmentSpec) -> int:
    return sum(int(x.nbytes) for x in jax.tree.leaves(_item_template(spec)))


def max_size_from_memory(spec: EnvironmentSpec, num_envs: int) -> int:
    """Number of steps that fit in the smaller of half the free RAM and one eighth of total RAM (rounded to a multiple of num_envs)."""
    memory = psutil.virtual_memory()
    budget = min(memory.available // 2, memory.total // 8)
    capacity = (budget // max(item_bytes(spec), 1) // num_envs) * num_envs
    return min(max(capacity, num_envs), MAX_AUTO_SIZE)


class Buffer:
    """Each env is one time-axis row. Steps are written in chunks of `sequence_length`.

    `sample` returns None until `min_size` transitions (summed over envs) have been added, so the learner
    doesn't start by overfitting a handful of early steps.

    `add` is called from the actor thread, `sample` from the learner thread. `add` donates the old state
    (to avoid copying the whole buffer), so every place that reads the state holds the lock. Fill level
    is tracked with a Python counter, so `size` / `can_sample` do not sync with the device.
    """

    def __init__(
        self,
        spec: EnvironmentSpec,
        *,
        num_envs: int,
        max_size: int | None,
        sample_batch_size: int,
        sequence_length: int,
        period: int = 1,
        min_size: int = 0,
    ):
        if not max_size or max_size <= 0:
            max_size = max_size_from_memory(spec, num_envs)
        print(f"replay max_size={max_size} (~{max_size * item_bytes(spec) / 1024**3:.1f} GB)")
        self._num_envs = num_envs
        self._max_length = max_size // num_envs
        self._buffer = flashbax.make_trajectory_buffer(
            add_batch_size=num_envs,
            sample_batch_size=sample_batch_size,
            sample_sequence_length=sequence_length,
            period=period,
            min_length_time_axis=sequence_length,  # Passing 0 printed needless logs.
            max_length_time_axis=self._max_length,
        )
        self._add = jax.jit(self._buffer.add, donate_argnums=0)
        self._sample = jax.jit(self._buffer.sample)
        self._state = self._buffer.init(_item_template(spec))
        self._lock = threading.Lock()
        self._chunk_size = sequence_length
        self._chunk: list[Transition] = []
        self._added_length = 0  # total steps written to each env row
        # Per-row steps needed before sampling; at least one chunk (flashbax can_sample), at most a full buffer.
        self._min_length = min(max(-(-min_size // num_envs), sequence_length), self._max_length)

    def add(self, transition: Transition) -> None:
        """Adds a single step shaped (num_envs, ...); writes to the buffer once a chunk is full."""
        self._chunk.append(transition)
        if len(self._chunk) < self._chunk_size:
            return
        chunk = jax.tree.map(lambda *steps: np.stack(steps, axis=1), *self._chunk)  # (num_envs, T, ...)
        self._chunk = []
        with self._lock:
            self._state = self._add(self._state, chunk)
            self._added_length += self._chunk_size

    def sample(self, key: jax.Array):
        """Returns None if no sample is available (buffer not yet full enough)."""
        with self._lock:
            if self._added_length < self._min_length:
                return None
            return self._sample(self._state, key)

    @property
    def size(self) -> int:
        return self._length * self._num_envs

    @property
    def fill_ratio(self) -> float:
        return self._length / self._max_length

    @property
    def _length(self) -> int:
        return min(self._added_length, self._max_length)
