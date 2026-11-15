"""Explicit recursive simulation with most-recent-first initial states."""

from dataclasses import dataclass

import numpy as np

from .models import ARSpecification, MASpecification


@dataclass(frozen=True)
class SimulationConfig:
    length: int = 200
    burn_in: int = 0
    seed: int = 20261115

    def __post_init__(self):
        for name, minimum in (("length", 1), ("burn_in", 0), ("seed", 0)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass
class SimulationResult:
    specification: ARSpecification | MASpecification
    config: SimulationConfig
    initial_state: tuple[float, ...]
    values: np.ndarray
    innovations: np.ndarray

    @property
    def retained_innovations(self):
        return self.innovations[self.config.burn_in :]


class ARSimulator:
    def __init__(self, specification: ARSpecification):
        self.specification = specification

    def path(self, innovations, initial_state):
        """Initial state is (y[-1], y[-2], ...); the first output is y[0]."""
        noise = np.asarray(innovations, dtype=float)
        history = np.asarray(initial_state, dtype=float).copy()
        if noise.ndim != 1 or not np.isfinite(noise).all():
            raise ValueError("Innovations must be a finite vector")
        if history.shape != (self.specification.order,) or not np.isfinite(history).all():
            raise ValueError("Initial state must have one finite value per lag")
        phi = np.asarray(self.specification.coefficients)
        output = np.empty(noise.size)
        with np.errstate(over="raise", invalid="raise"):
            for t, innovation in enumerate(noise):
                value = self.specification.intercept + phi @ history + innovation
                output[t] = value
                history[1:] = history[:-1]
                history[0] = value
        if not np.isfinite(output).all():
            raise FloatingPointError("Simulation exceeded finite numerical range")
        return output

    def simulate(self, config: SimulationConfig, initial_state, innovations=None):
        if innovations is None:
            innovations = np.random.default_rng(config.seed).normal(
                scale=np.sqrt(self.specification.innovation_variance),
                size=config.length + config.burn_in,
            )
        noise = np.asarray(innovations, dtype=float)
        if noise.shape != (config.length + config.burn_in,):
            raise ValueError("Supply exactly length + burn_in innovations")
        full = self.path(noise, initial_state)
        return SimulationResult(
            self.specification, config, tuple(initial_state), full[config.burn_in :], noise.copy()
        )


class MASimulator:
    """Finite moving averages; initial_state stores past innovations, most recent first."""

    def __init__(self, specification: MASpecification):
        self.specification = specification

    def simulate(self, config: SimulationConfig, initial_state, innovations=None):
        if innovations is None:
            innovations = np.random.default_rng(config.seed).normal(
                scale=np.sqrt(self.specification.innovation_variance),
                size=config.length + config.burn_in,
            )
        noise = np.asarray(innovations, dtype=float)
        history = np.asarray(initial_state, dtype=float).copy()
        if noise.shape != (config.length + config.burn_in,) or not np.isfinite(noise).all():
            raise ValueError("Supply length + burn_in finite innovations")
        if history.shape != (self.specification.order,) or not np.isfinite(history).all():
            raise ValueError("Supply one finite initial innovation per MA lag")
        coefficients = np.asarray(self.specification.coefficients)
        output = np.empty(noise.size)
        for t, value in enumerate(noise):
            output[t] = value + coefficients[1:] @ history
            if history.size:
                history[1:] = history[:-1]
                history[0] = value
        if not np.isfinite(output).all():
            raise FloatingPointError("MA simulation exceeded finite numerical range")
        return SimulationResult(
            self.specification, config, tuple(initial_state), output[config.burn_in :], noise.copy()
        )
