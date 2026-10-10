"""Stochastic MuZero and Gumbel MuZero, built on JAX, mctx and Gymnasium.

Module map:
    config.py      MuZeroConfig
    algorithm.py   Algorithm, SearchPolicy: the contract between the infrastructure and an algorithm
    algorithms/    ALGORITHMS registry and the algorithms
        muzero.py      shared math of the MuZero family: Support, network blocks, loss skeleton
        smz.py / gmz.py  algorithms (config, networks, search, loss)
    replay.py / reanalyze.py  flashbax replay and reanalyze iterator
    actor.py / learner.py / loop.py  environment interaction, gradient step, env loop
    observers.py / loggers.py / checkpoint.py  metrics, terminal/W&B writers, orbax checkpoint
    experiment.py  ExperimentConfig, run_experiment, load_actor
"""

from mujax.actor import Actor
from mujax.algorithm import Algorithm
from mujax.algorithms import ALGORITHMS, GMZ, SMZ, GMZConfig, SMZConfig, algorithm_for
from mujax.checkpoint import CheckpointingConfig
from mujax.config import MuZeroConfig
from mujax.experiment import EvaluationConfig, ExperimentConfig, load_actor, run_experiment
from mujax.learner import Learner
from mujax.loggers import ConsoleWriter, Logger, WandbWriter, Writer, create_writer
from mujax.loop import EnvironmentLoop
from mujax.observers import EnvLoopObserver, InfoKeysObserver
from mujax.types import EnvironmentSpec, as_vector_env, make_environment_spec

__version__ = "0.1.0"

__all__ = [
    "ALGORITHMS",
    "GMZ",
    "SMZ",
    "Actor",
    "Algorithm",
    "CheckpointingConfig",
    "ConsoleWriter",
    "EnvLoopObserver",
    "EnvironmentLoop",
    "EnvironmentSpec",
    "EvaluationConfig",
    "ExperimentConfig",
    "GMZConfig",
    "InfoKeysObserver",
    "Learner",
    "Logger",
    "MuZeroConfig",
    "SMZConfig",
    "WandbWriter",
    "Writer",
    "algorithm_for",
    "as_vector_env",
    "create_writer",
    "load_actor",
    "make_environment_spec",
    "run_experiment",
]
