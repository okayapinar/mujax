"""MuZero, Stochastic MuZero, Gumbel MuZero and Sampled MuZero, built on JAX, mctx and Gymnasium.

Module map:
    config.py      MuZeroConfig
    algorithm.py   Algorithm, SearchPolicy: the contract between the infrastructure and an algorithm
    algorithms/    ALGORITHMS registry and the algorithms
        muzero.py      shared math of the MuZero family: Support, network blocks, loss skeleton, PUCT exploration
        mz.py / smz.py / gmz.py / sampled_mz.py  algorithms (config, networks, search, loss); gmz and sampled_mz reuse the mz model
    replay.py / reanalyze.py  flashbax replay and reanalyze iterator
    actor.py / learner.py / loop.py  environment interaction, gradient step, env loop
    observers.py / loggers.py / checkpoint.py  metrics, CLU metric writers (TensorBoard/console/W&B), orbax checkpoint
    experiment.py  ExperimentConfig, run_experiment, load_actor
"""

from mujax.actor import Actor
from mujax.algorithm import Algorithm
from mujax.algorithms import ALGORITHMS, GMZ, MZ, SMZ, GMZConfig, MZConfig, SMZConfig, SampledMZ, SampledMZConfig, algorithm_for
from mujax.checkpoint import CheckpointingConfig
from mujax.config import MuZeroConfig
from mujax.experiment import EvaluationConfig, ExperimentConfig, load_actor, run_experiment
from mujax.learner import Learner
from mujax.loggers import Logger, WandbWriter, create_writer
from mujax.loop import EnvironmentLoop
from mujax.observers import EnvLoopObserver, InfoKeysObserver
from mujax.types import EnvironmentSpec, make_environment_spec

__version__ = "0.1.0"

__all__ = [
    "ALGORITHMS",
    "GMZ",
    "SMZ",
    "Actor",
    "Algorithm",
    "CheckpointingConfig",
    "EnvLoopObserver",
    "EnvironmentLoop",
    "EnvironmentSpec",
    "EvaluationConfig",
    "ExperimentConfig",
    "GMZConfig",
    "InfoKeysObserver",
    "MZ",
    "MZConfig",
    "Learner",
    "Logger",
    "MuZeroConfig",
    "SMZConfig",
    "SampledMZ",
    "SampledMZConfig",
    "WandbWriter",
    "algorithm_for",
    "create_writer",
    "load_actor",
    "make_environment_spec",
    "run_experiment",
]
