# mujax

MuZero, Stochastic MuZero and Gumbel MuZero for [Gymnasium](https://gymnasium.farama.org/) environments, built on
JAX, [mctx](https://github.com/google-deepmind/mctx) and [flashbax](https://github.com/instadeepai/flashbax).

- **Three algorithms, one pipeline**: `MZ` (MuZero with PUCT search), `SMZ` (Stochastic MuZero with afterstates and a
  VQ chance codebook) and `GMZ` (Gumbel MuZero) share replay, reanalyze, actor, learner and checkpointing.
- **Single process**: actor and evaluator run in threads on CPU, the learner runs on GPU when available.
- **Reanalyze**: a fraction of every batch is re-searched with the latest parameters.
- **Categorical value/reward** with HL-Gauss targets in symlog space.
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
mujax --env CartPole-v1 --num-steps 20000 --no-wandb gmz --num-simulations 16
mujax --env Acrobot-v1 smz --batch-size 1024
mujax --env CartPole-v1 mz --dirichlet-fraction 0.25
mujax --help          # all options; `mujax mz --help`, `mujax smz --help`, `mujax gmz --help` for algorithm options
```

Checkpoints are written to `checkpoints/<algo>_<timestamp>/`.

## Python API

```python
import gymnasium as gym
from mujax import CheckpointingConfig, ExperimentConfig, GMZConfig, load_actor, run_experiment


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

actor, metadata = load_actor("checkpoints/cartpole")
```

Environments must use `AutoresetMode.SAME_STEP`. See [`examples/`](https://github.com/okayapinar/mujax/tree/master/examples) for complete training and evaluation scripts.

## Layout

| Module | Contents |
| --- | --- |
| `config.py` | `MuZeroConfig`, shared hyperparameters |
| `algorithm.py` | `Algorithm`, `SearchPolicy`: the contract between the infrastructure and an algorithm |
| `algorithms/__init__.py` | `ALGORITHMS` registry, `algorithm_for` |
| `algorithms/muzero.py` | shared math of the MuZero family: `Support`, network blocks, loss skeleton, PUCT exploration |
| `algorithms/mz.py`, `algorithms/smz.py`, `algorithms/gmz.py` | algorithms: config, networks, search, loss; `gmz` reuses the `mz` model |
| `replay.py`, `reanalyze.py` | flashbax replay and reanalyze iterator |
| `actor.py`, `learner.py`, `loop.py` | environment interaction, gradient step, environment loop |
| `observers.py`, `loggers.py`, `checkpoint.py` | metrics, logging, Orbax checkpoints |
| `experiment.py` | `ExperimentConfig`, `run_experiment`, `load_actor` |

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
