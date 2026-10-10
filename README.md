# mujax

Gumbel MuZero and Stochastic MuZero for [Gymnasium](https://gymnasium.farama.org/) environments, built on
JAX, [mctx](https://github.com/google-deepmind/mctx) and [flashbax](https://github.com/instadeepai/flashbax).

- **Two algorithms, one pipeline**: `GMZ` (Gumbel MuZero) and `SMZ` (Stochastic MuZero with afterstates and a
  VQ chance codebook) share replay, reanalyze, actor, learner and checkpointing.
- **Single process**: actors and evaluator run in threads on CPU, the learner runs on GPU when available. The CPU
  search releases the GIL, so `num_actors` actor threads (`--num-actors`) scale env throughput with the core count.
- **Reanalyze**: a fraction of every batch is re-searched with the latest parameters.
- **Categorical value/reward** with HL-Gauss targets in symlog space.
- **Symlog observations**: raw observations enter the networks through `signed_logp1` (DreamerV3), so no per-dimension normalization is needed.
- **Checkpoints**: best parameters by evaluation score plus full training state for resuming (Orbax), optional W&B upload.

The design is inspired by DeepMind's [Acme](https://github.com/google-deepmind/acme): the split into actor, learner,
replay and environment loop, the observers and the counter follow Acme's structure, reduced to a small single-process
library.

Requirements: 1-D `Box` observations and `Discrete` actions (use `gym.wrappers.FlattenObservation` otherwise).
Illegal actions can be masked through `info["invalid_actions"]`.

## Installation

```bash
pip install mujax                 # CPU
pip install "mujax[cuda12]"       # NVIDIA GPU
pip install "mujax[wandb]"        # Weights & Biases logging
```

## Command line

```bash
mujax --env CartPole-v1 --num-steps 20000 gmz --num-simulations  16
mujax --env Acrobot-v1 smz --batch-size 1024
mujax --help          # all options; `mujax gmz --help`, `mujax smz --help` for algorithm options
```

Checkpoints are written to `checkpoints/<algo>_<timestamp>/`.

## Python API

```python
import gymnasium as gym
from mujax import CheckpointingConfig, ExperimentConfig, GMZConfig, load_policy, run_experiment


def make_env(num_envs):
    return gym.make_vec(
        "CartPole-v1", num_envs=num_envs, vectorization_mode="sync", vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
    )


run_experiment(
    ExperimentConfig(
        config=GMZConfig(batch_size=256).with_num_steps(20_000),
        environment_factory=lambda seed: make_env(16),
        eval_environment_factory=lambda seed: make_env(1),
        max_num_learner_steps=20_000,
        checkpointing=CheckpointingConfig(directory="checkpoints/cartpole"),
    )
)

policy = load_policy("checkpoints/cartpole")
action = policy.act(observation)  # int for one observation, (N,) array for a batch
```

Environment factories may return a single `gym.Env` (used as `num_envs=1`) or a vector env; vector envs must use `AutoresetMode.SAME_STEP`. See [`examples/`](https://github.com/okayapinar/mujax/tree/master/examples) for training scripts (classic control, Pendulum with discrete torques, FrozenLake, Blackjack) and evaluation (`evaluate.py`).

### Checkpoints

A run writes two checkpoints under `CheckpointingConfig.directory` (`checkpoints/<algo>_<timestamp>/` on the command line):

| Directory | Contents | Used by |
| --- | --- | --- |
| `best/` | params with the best evaluator score (EMA) | `load_policy`, `load_actor` |
| `latest/` | full learner state: params, optimizer, step | the next run in the same directory (`resume=True`) |

Each is written at most once per `every_sec` (default 30 minutes; `0` writes only at the end of the run) and replaced atomically. A resumed run keeps the previous best score, so `best/` is only overwritten by better params. With an active W&B run, every `best/` write is uploaded as a `model` artifact (`upload_to_wandb`).

For inference, `load_policy` takes a run directory, a `best/` directory or a W&B artifact reference (`entity/project/run-checkpoint:best`) and returns a `Policy`:

```python
policy = load_policy("checkpoints/cartpole")
env = gym.make(policy.metadata["extra"]["env_id"])   # `extra` is `ExperimentConfig.extra` of the training run
observation, _ = env.reset()
action = policy.act(observation)                      # single observation -> int
actions = policy.act(observations)                    # (N, obs_dim) batch -> (N,)
output = policy.search(observations)                  # action, policy_probs and value of the search
actor = policy.actor()                                # batched `Actor` for `EnvironmentLoop`
```

`policy.act(observation, invalid_actions=mask)` masks actions (1 = invalid), like `info["invalid_actions"]` during training. The search is jitted per batch size.

### Logging

`create_writer` sends metrics to the terminal and, optionally, Weights & Biases; learner, actor and evaluator metrics are written under `learner/`, `actor/` and `evaluator/` with `learner_steps` as the step:

```python
writer = create_writer(console=True, wandb_project="mujax")
ExperimentConfig(..., writer=writer)
```

Any object with `write(step, scalars)`, `write_config(config)` and `close()` works as a writer (`mujax.Writer`). Without a writer only the progress bar is shown.

On the command line metrics are printed to the terminal (`--no-console` turns it off); add `--wandb` (`--wandb-project`) for W&B.

## Layout

| Module | Contents |
| --- | --- |
| `config.py` | `MuZeroConfig`, shared hyperparameters |
| `algorithm.py` | `Algorithm`, `SearchPolicy`: the contract between the infrastructure and an algorithm |
| `algorithms/__init__.py` | `ALGORITHMS` registry, `algorithm_for` |
| `algorithms/muzero.py` | shared math of the MuZero family: `Support`, network blocks, loss skeleton |
| `algorithms/smz.py`, `algorithms/gmz.py` | algorithms: config, networks, search, loss |
| `replay.py`, `reanalyze.py` | flashbax replay and reanalyze iterator |
| `actor.py`, `learner.py`, `loop.py` | environment interaction, gradient step, environment loop |
| `observers.py`, `loggers.py`, `checkpoint.py` | metrics, terminal and W&B writers, `best/` and `latest/` checkpoints (Orbax) |
| `experiment.py` | `ExperimentConfig`, `run_experiment` |
| `inference.py` | `Policy`, `load_policy`, `load_actor`: running a checkpoint outside of training |

## Development

```bash
uv sync
uv run pytest
```

## References

- Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (2020)
- Antonoglou et al., *Planning in Stochastic Environments with a Learned Model* (2022)
- Danihelka et al., *Policy Improvement by Planning with Gumbel* (2022)
- Hoffman et al., *Acme: A Research Framework for Distributed Reinforcement Learning* (2020)

## License

Apache-2.0
