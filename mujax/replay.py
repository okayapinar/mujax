"""Flashbax-based, thread-safe trajectory replay."""

from __future__ import annotations

import collections
import threading

import flashbax
import jax
import jax.numpy as jnp
import numpy as np
import psutil

from mujax.types import EnvironmentSpec, Transition

MAX_AUTO_SIZE = 5_000_000
_RATIO_TOLERANCE = 0.1  # SampleToInsertRatio slack as a fraction of the samples expected at min_size (Acme's default)
_WAIT_SEC = 0.05  # how often a blocked insert/sample re-checks the stop event


class SampleToInsertRatio:
    """Keeps `samples ~= samples_per_insert * (inserts - min_size)`; Reverb's rate limiter of the same name.

    `insert` blocks while the actor is too far ahead (error > error_buffer), `sample` blocks while the learner is
    too far ahead (error < -error_buffer); the two conditions are disjoint, so they cannot block each other.
    Counting starts at `min_size`, so the first samples are allowed as soon as the buffer is ready to sample.
    Waiters re-check `stop_event` periodically and return without blocking once it is set.
    """

    def __init__(self, samples_per_insert: float, *, min_size: int, error_buffer: float, stop_event: threading.Event | None = None):
        if samples_per_insert <= 0:
            raise ValueError(f"samples_per_insert pozitif olmali, gelen: {samples_per_insert}")
        self._ratio = float(samples_per_insert)
        self._min_size = int(min_size)
        self._error_buffer = float(error_buffer)
        self._stop_event = stop_event
        self._inserts = 0
        self._samples = 0
        self._condition = threading.Condition()

    @property
    def error(self) -> float:
        """Samples expected so far minus samples taken; positive: the actor is ahead."""
        return self._ratio * (self._inserts - self._min_size) - self._samples

    def insert(self, num_items: int) -> None:
        self._wait(lambda: self.error <= self._error_buffer)
        with self._condition:
            self._inserts += int(num_items)
            self._condition.notify_all()

    def sample(self, num_items: int) -> None:
        self._wait(lambda: self.error >= -self._error_buffer)
        with self._condition:
            self._samples += int(num_items)
            self._condition.notify_all()

    def _wait(self, allowed) -> None:
        with self._condition:
            while not allowed():
                if self._stop_event is not None and self._stop_event.is_set():
                    return
                self._condition.wait(_WAIT_SEC)


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


class Adder:
    """One actor's write handle: collects steps shaped (num_envs, ...) and hands full chunks to the buffer.

    Created by `Buffer.adder`; each actor thread needs its own instance (the chunk is not shared).
    """

    def __init__(self, buffer: Buffer, index: int) -> None:
        self._buffer = buffer
        self._index = index
        self._chunk: list[Transition] = []

    def add(self, transition: Transition) -> None:
        """Adds a single step shaped (num_envs, ...); writes to the buffer once a chunk is full."""
        self._chunk.append(transition)
        if len(self._chunk) < self._buffer.chunk_size:
            return
        chunk = jax.tree.map(lambda *steps: np.stack(steps, axis=1), *self._chunk)  # (num_envs, T, ...)
        self._chunk = []
        self._buffer.write(self._index, chunk)


class Buffer:
    """Each env is one time-axis row. Steps are written in chunks of `sequence_length`.

    `sample` returns None until `min_size` transitions (summed over envs) have been added, so the learner
    doesn't start by overfitting a handful of early steps.

    `add` is called from the actor thread, `sample` from the learner thread. `add` donates the old state
    (to avoid copying the whole buffer), so every place that reads the state holds the lock. Fill level
    is tracked with a Python counter, so `size` / `can_sample` do not sync with the device.

    With `num_actors > 1` each actor thread writes through its own `adder(i)`; the buffer has
    `num_actors * num_envs` rows and a chunk is written only once every actor has delivered one, so the
    chunks of a row stay contiguous in time (a sequence never crosses from one actor's trajectory into
    another's). Chunks are concatenated in actor order; actor i owns rows i*num_envs..(i+1)*num_envs-1.

    With `replay_ratio` (sequences sampled per transition inserted) a `SampleToInsertRatio` limiter makes `add`
    and `sample` wait for each other; `stop_event` releases the waiters at the end of the run.
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
        replay_ratio: float | None = None,
        stop_event: threading.Event | None = None,
        num_actors: int = 1,
    ):
        if num_actors < 1:
            raise ValueError(f"num_actors pozitif olmali, gelen: {num_actors}")
        num_rows = num_envs * num_actors
        if not max_size or max_size <= 0:
            max_size = max_size_from_memory(spec, num_rows)
        print(f"replay max_size={max_size} (~{max_size * item_bytes(spec) / 1024**3:.1f} GB)")
        self._num_envs = num_envs
        self._num_rows = num_rows
        self._sample_batch_size = sample_batch_size
        self._max_length = max_size // num_rows
        self._buffer = flashbax.make_trajectory_buffer(
            add_batch_size=num_rows,
            sample_batch_size=sample_batch_size,
            sample_sequence_length=sequence_length,
            period=period,
            min_length_time_axis=sequence_length,  # Passing 0 printed needless logs.
            max_length_time_axis=self._max_length,
        )
        self._add = jax.jit(self._buffer.add, donate_argnums=0)
        self._sample = jax.jit(self._buffer.sample)
        self._state = self._buffer.init(_item_template(spec))
        self._lock = threading.Lock()  # guards `_state` and `_added_length`
        # Serializes chunk writes (so chunks land on the time axis in delivery order) and the per-actor queues.
        self._write_lock = threading.Lock()
        self.chunk_size = sequence_length
        self._adders = [Adder(self, index) for index in range(num_actors)]
        self._pending: list[collections.deque] = [collections.deque() for _ in range(num_actors)]
        self._added_length = 0  # total steps written to each env row
        # Per-row steps needed before sampling; at least one chunk (flashbax can_sample), at most a full buffer.
        self._min_length = min(max(-(-min_size // num_rows), sequence_length), self._max_length)
        self._limiter = None
        if replay_ratio is not None:
            # Inserts are counted when a chunk is written, so "actor ahead" implies the buffer is ready to sample.
            min_size_to_sample = self._min_length * num_rows
            error_buffer = max(float(sample_batch_size), _RATIO_TOLERANCE * replay_ratio * min_size_to_sample)
            self._limiter = SampleToInsertRatio(replay_ratio, min_size=min_size_to_sample, error_buffer=error_buffer, stop_event=stop_event)

    @property
    def num_actors(self) -> int:
        return len(self._adders)

    def adder(self, index: int) -> Adder:
        """The write handle of actor `index` (0 <= index < num_actors)."""
        return self._adders[index]

    def add(self, transition: Transition) -> None:
        """Single-actor shortcut for `adder(0).add`."""
        if self.num_actors != 1:
            raise RuntimeError(f"num_actors={self.num_actors}: her aktor kendi `adder(i)` uzerinden yazmali")
        self._adders[0].add(transition)

    def write(self, index: int, chunk: Transition) -> None:
        """Queues actor `index`'s chunk shaped (num_envs, T, ...) and writes one combined chunk once every actor has one.

        Called by `Adder`. May wait for the learner (`replay_ratio`); while it waits, the other actors block here too,
        which is the limiter's intent (the actors are ahead).
        """
        with self._write_lock:
            self._pending[index].append(chunk)
            if not all(self._pending):
                return
            chunks = [queue.popleft() for queue in self._pending]
            combined = chunks[0] if len(chunks) == 1 else jax.tree.map(lambda *rows: np.concatenate(rows, axis=0), *chunks)
            if self._limiter is not None:
                self._limiter.insert(self.chunk_size * self._num_rows)
            with self._lock:
                self._state = self._add(self._state, combined)
                self._added_length += self.chunk_size

    def sample(self, key: jax.Array):
        """Returns None if no sample is available (buffer not yet full enough); may wait for the actor (`replay_ratio`)."""
        with self._lock:
            if self._added_length < self._min_length:
                return None
        if self._limiter is not None:
            self._limiter.sample(self._sample_batch_size)
        with self._lock:
            return self._sample(self._state, key)

    @property
    def size(self) -> int:
        return self._length * self._num_rows

    @property
    def fill_ratio(self) -> float:
        return self._length / self._max_length

    @property
    def _length(self) -> int:
        return min(self._added_length, self._max_length)
