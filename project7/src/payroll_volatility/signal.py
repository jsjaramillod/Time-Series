"""Known-parameter decomposition of an observed message into signal and noise."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov, toeplitz
from scipy.stats import norm

MESSAGE_SHA = "c58aaeda337f520b34d2bba42248b0c2c1ce7368aae7366872703a4b2421ab1b"


def signal_control(path):
    file = Path(path)
    if hashlib.sha256(file.read_bytes()).hexdigest() != MESSAGE_SHA:
        raise ValueError("Unexpected message snapshot")
    data = pd.read_csv(file)
    if data.month.tolist() != list(range(1, 201)):
        raise ValueError("Expected 200 consecutive messages")
    return {**filter_messages(data.message.to_numpy()), "source_sha256": MESSAGE_SHA}


def filter_messages(values):
    """Condition on known-parameter messages recursively and in a single Gaussian block."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a finite, nonempty series")
    f = np.array([[0.5, 0.4, 0], [1.0, 0, 0], [0, 0, 0.7]])
    q = np.diag([1.0, 0.0, 0.5])
    h = np.array([1.0, 0.0, 1.0])
    stationary = solve_discrete_lyapunov(f, q)
    mean, cov = np.zeros(3), stationary.copy()
    for obs in values:
        mean, cov = f @ mean, f @ cov @ f.T + q
        cross = cov @ h
        gain = cross / (h @ cross)
        mean += gain * (obs - h @ mean)
        cov -= np.outer(gain, cross)
    # Independent full-vector Gaussian conditioning in chronological order.
    powers = [np.linalg.matrix_power(f, j) @ stationary for j in range(len(values))]
    joint = toeplitz([h @ g @ h for g in powers])
    cross = np.array([g[0] @ h for g in powers[::-1]])
    weights = np.linalg.solve(joint, cross)
    np.testing.assert_allclose(
        [mean[0], cov[0, 0]],
        [weights @ values, stationary[0, 0] - weights @ cross],
        atol=1e-10,
        rtol=0,
    )
    quantiles = norm.ppf([0.25, 0.5, 0.75], mean[0], np.sqrt(cov[0, 0]))
    return dict(
        n=len(values),
        last_message=values[-1],
        mean=mean[0],
        variance=cov[0, 0],
        q25=quantiles[0],
        median=quantiles[1],
        q75=quantiles[2],
        max_conditioning_difference=float(
            max(
                abs(mean[0] - weights @ values), abs(cov[0, 0] - stationary[0, 0] + weights @ cross)
            )
        ),
    )
