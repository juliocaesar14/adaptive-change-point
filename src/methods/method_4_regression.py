# Method 4, regression on related series.
# Predict the target from related series using recent data only.
# The structure adapts because the most correlated series are chosen again daily.
# The settings adapt because the coefficients are fitted again daily.
# An alarm means the target has broken away from its usual relationship.
# It does not mean the market has changed direction.

import numpy as np


def regression_gap(y, related, length, top_k):
    # Returns the gap in units of recent model error for each day.
    y = np.asarray(y, dtype=float)
    related = np.asarray(related, dtype=float)
    n, m = len(y), related.shape[1]
    k = min(top_k, m)
    gap = np.full(n, np.nan)
    returns_y = np.diff(y, prepend=y[0])
    returns_r = np.diff(related, axis=0, prepend=related[:1])
    for t in range(length + 1, n):
        # Use only days before today for choosing and fitting.
        lo, hi = t - length, t
        ry = returns_y[lo:hi] - returns_y[lo:hi].mean()
        rr = returns_r[lo:hi] - returns_r[lo:hi].mean(axis=0)
        size = np.sqrt((rr * rr).sum(axis=0) * (ry * ry).sum()) + 1e-12
        corr = (rr * ry[:, None]).sum(axis=0) / size
        chosen = np.argpartition(-corr, k - 1)[:k]
        # Ordinary least squares with a constant, Equation 13 in the paper.
        design = np.column_stack((np.ones(length), related[lo:hi][:, chosen]))
        beta, *_ = np.linalg.lstsq(design, y[lo:hi], rcond=None)
        residual = y[lo:hi] - design @ beta
        spread = residual.std(ddof=k + 1) + 1e-12
        prediction = beta[0] + related[t, chosen] @ beta[1:]
        gap[t] = (prediction - y[t]) / spread
    return gap


def gap_alarms(gap, threshold):
    # Fire when the gap crosses the limit, Equation 16 in the paper.
    # Wait until the gap falls below half the limit before firing again.
    alarms = []
    armed = True
    for t, value in enumerate(gap):
        if np.isnan(value):
            continue
        if armed and abs(value) > threshold:
            alarms.append((t, 1 if value > 0 else -1))
            armed = False
        elif abs(value) < threshold / 2:
            armed = True
    return alarms


def regression_alarms(y, related, length, top_k, threshold):
    # Join the two steps above.
    return gap_alarms(regression_gap(y, related, length, top_k), threshold)
