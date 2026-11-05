# Fake markets where the true change dates are known.
# Case A, up and down trends with calm Gaussian noise.
# Case B, the same trends with changing volatility and fat tails.
# Case C, the target leaves and rejoins related series.

import numpy as np


def _segment_starts(rng, days, low, high):
    # Pick random segment lengths until the series is full.
    starts, current = [], 0
    while True:
        current += int(rng.integers(low, high + 1))
        if current >= days - low // 2:
            return np.array(starts, dtype=int)
        starts.append(current)


def _noise(rng, days, heavy):
    # Calm Gaussian noise, or Student t noise with changing volatility.
    if not heavy:
        return rng.normal(0.0, 0.01, days)
    # Student t with 4 degrees of freedom has variance 2, so divide by root 2.
    shocks = rng.standard_t(4, days) / np.sqrt(2.0)
    level = np.empty(days)
    current, position = 0.01, 0
    while position < days:
        span = int(rng.integers(100, 300))
        level[position:position + span] = current
        current = float(rng.choice([0.006, 0.01, 0.02]))
        position += span
    return shocks * level


def _related(rng, factor, m=4):
    # Related series share the common factor and have small private noise.
    private = rng.normal(0.0, 0.003, (len(factor), m))
    return np.cumsum(factor[:, None] + private, axis=0)


def simulate_trend(seed, days=1500, heavy=False):
    # Direction flips at each planted date and the size varies by segment.
    rng = np.random.default_rng(seed)
    truth = _segment_starts(rng, days, 120, 300)
    edges = np.concatenate(([0], truth, [days]))
    drift = np.empty(days)
    sign = 1.0
    for a, b in zip(edges[:-1], edges[1:]):
        drift[a:b] = sign * rng.uniform(0.002, 0.004)
        sign = -sign
    returns = drift + _noise(rng, days, heavy)
    return {"y": np.cumsum(returns), "related": _related(rng, returns), "truth": truth}


def simulate_relationship_break(seed, days=1500):
    # The market factor has no planted reversals here.
    # During a break the target also gains its own steady drift.
    rng = np.random.default_rng(seed)
    truth = _segment_starts(rng, days, 120, 300)
    edges = np.concatenate(([0], truth, [days]))
    factor = rng.normal(0.0003, 0.01, days)
    extra = np.zeros(days)
    breaking = False
    for a, b in zip(edges[:-1], edges[1:]):
        if breaking:
            extra[a:b] = rng.choice([-1.0, 1.0]) * rng.uniform(0.002, 0.004)
        breaking = not breaking
    returns = factor + extra + rng.normal(0.0, 0.003, days)
    return {"y": np.cumsum(returns), "related": _related(rng, factor), "truth": truth}


CASES = {
    "A clean trends": lambda seed, days: simulate_trend(seed, days, False),
    "B wild noise": lambda seed, days: simulate_trend(seed, days, True),
    "C relationship breaks": simulate_relationship_break,
}
