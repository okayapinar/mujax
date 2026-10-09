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


def test_sample_waits_for_min_size():
    import jax

    # min_size=25 over 2 envs -> 13 steps per row -> sampling starts after the third chunk of 5 steps.
    buffer = Buffer(SPEC, num_envs=2, max_size=64, sample_batch_size=4, sequence_length=5, min_size=25)
    key = jax.random.PRNGKey(0)
    for _ in range(10):
        buffer.add(transition(2))
    assert buffer.size == 20 and buffer.sample(key) is None
    for _ in range(5):
        buffer.add(transition(2))
    assert buffer.sample(key) is not None


def test_min_size_is_capped_at_max_size():
    import jax

    buffer = Buffer(SPEC, num_envs=2, max_size=20, sample_batch_size=4, sequence_length=5, min_size=1000)
    for _ in range(10):
        buffer.add(transition(2))
    assert buffer.sample(jax.random.PRNGKey(0)) is not None


def test_rate_limiter_blocks_the_side_that_is_ahead():
    import threading

    from mujax.replay import SampleToInsertRatio

    limiter = SampleToInsertRatio(2.0, min_size=10, error_buffer=4.0)
    limiter.insert(10)  # reaches min_size: error 0
    limiter.insert(2)  # error 4: at the buffer, the next call still passes (the check happens before counting)
    limiter.insert(2)  # error 8
    assert limiter.error == 8.0

    # Past the buffer: the next insert waits until the learner samples enough.
    done = threading.Event()
    threading.Thread(target=lambda: (limiter.insert(2), done.set()), daemon=True).start()
    assert not done.wait(0.2)
    limiter.sample(8)  # error 0 -> the insert goes through -> error 4
    assert done.wait(2.0)
    assert limiter.error == 4.0

    # Symmetric: sampling past the buffer waits for an insert.
    limiter.sample(4)  # 0
    limiter.sample(4)  # -4
    limiter.sample(4)  # -8
    done.clear()
    threading.Thread(target=lambda: (limiter.sample(4), done.set()), daemon=True).start()
    assert not done.wait(0.2)
    limiter.insert(4)  # 0 -> the sample goes through -> -4
    assert done.wait(2.0)
    assert limiter.error == -4.0


def test_rate_limiter_releases_waiters_on_stop():
    import threading

    from mujax.replay import SampleToInsertRatio

    stop = threading.Event()
    limiter = SampleToInsertRatio(1.0, min_size=0, error_buffer=0.0, stop_event=stop)
    limiter.insert(5)
    done = threading.Event()
    threading.Thread(target=lambda: (limiter.insert(5), done.set()), daemon=True).start()
    assert not done.wait(0.2)
    stop.set()
    assert done.wait(2.0)


def test_buffer_replay_ratio_keeps_actor_and_learner_in_step():
    """With replay_ratio the actor waits for the learner once it is ahead, and the learner never deadlocks."""
    import threading

    import jax

    stop = threading.Event()
    # min_size 10 (one chunk over 2 envs), batch 4, ratio 1: error_buffer = max(4, 0.1 * 10) = 4 samples.
    buffer = Buffer(SPEC, num_envs=2, max_size=200, sample_batch_size=4, sequence_length=5, replay_ratio=1.0, stop_event=stop)
    added = 0

    def actor():
        nonlocal added
        for _ in range(100):
            buffer.add(transition(2))
            added += 1

    thread = threading.Thread(target=actor, daemon=True)
    thread.start()
    thread.join(1.0)
    # Two chunks (20 transitions) are in, the third flush (at the 15th add) waits: 10 past min_size, nothing sampled.
    assert thread.is_alive() and added == 14

    key = jax.random.PRNGKey(0)
    samples = 0
    while thread.is_alive():
        assert buffer.sample(key) is not None
        samples += 1
        thread.join(0.01)
    # 200 transitions, counted from min_size 10 -> ~190 sampled sequences; the actor may lead by
    # error_buffer + one chunk (14), the learner by error_buffer (4).
    assert 176 <= samples * 4 <= 194
    stop.set()


def test_multi_actor_chunks_keep_rows_contiguous():
    """Each actor's chunks go to its own rows, and a chunk is written only once every actor has delivered one."""
    import jax

    def marked(num_envs: int, actor: int, step: int) -> Transition:
        t = transition(num_envs)
        obs = np.zeros((num_envs, SPEC.obs_dim), np.float32)
        obs[:, 0], obs[:, 1] = actor, step
        return t._replace(observation=obs)

    buffer = Buffer(SPEC, num_envs=2, max_size=200, sample_batch_size=16, sequence_length=5, num_actors=2)
    with pytest.raises(RuntimeError):
        buffer.add(transition(2))
    a0, a1 = buffer.adder(0), buffer.adder(1)
    for step in range(10):  # actor 0 is two chunks ahead; nothing lands until actor 1 catches up
        a0.add(marked(2, 0, step))
    assert buffer.size == 0
    for step in range(5):
        a1.add(marked(2, 1, step))
    assert buffer.size == 20  # one combined chunk: 4 rows x 5 steps
    for step in range(5, 10):
        a1.add(marked(2, 1, step))
    assert buffer.size == 40

    batch = buffer.sample(jax.random.PRNGKey(0)).experience.observation  # (16, 5, obs_dim)
    actor_ids, steps = np.asarray(batch[..., 0]), np.asarray(batch[..., 1])
    assert (actor_ids == actor_ids[:, :1]).all()  # a sequence never mixes two actors
    assert (np.diff(steps, axis=1) == 1).all()  # and walks the actor's own time axis in order
