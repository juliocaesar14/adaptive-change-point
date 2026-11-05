# Method 3, adaptive window.
# Fit a straight line to recent prices using least squares.
# After a large surprise the window shrinks, so old data is dropped.
# In calm times the window grows by one day, so the estimate gets steadier.
# Setting min length equal to max length gives the fixed window control.

import numpy as np
from src.detector import direction_flip_alarms


def _fit_line(s0, s1, s2, t, length):
    # Least squares using running sums, so each fit is fast.
    # Time is measured so that today is 0 and earlier days are negative.
    start = t - length + 1
    n = float(length)
    sum_y = s0[t + 1] - s0[start]
    sum_iy = s1[t + 1] - s1[start]
    sum_yy = s2[t + 1] - s2[start]
    sum_u = -n * (n - 1) / 2.0
    sum_uu = (n - 1) * n * (2 * n - 1) / 6.0
    # Index i equals u plus t, this converts the sum to the shifted time.
    sum_uy = sum_iy - t * sum_y
    bottom = n * sum_uu - sum_u * sum_u
    slope = (n * sum_uy - sum_u * sum_y) / bottom
    level = (sum_y - slope * sum_u) / n
    # Residual spread is used to judge if tomorrow is surprising.
    rss = sum_yy - level * sum_y - slope * sum_uy
    spread = np.sqrt(max(rss, 0.0) / max(n - 2, 1))
    return level, slope, spread


def adaptive_window_line(y, min_length, max_length, surprise=2.5):
    # Returns the line end value and the window length used each day.
    y = np.asarray(y, dtype=float)
    index = np.arange(len(y))
    s0 = np.concatenate(([0.0], np.cumsum(y)))
    s1 = np.concatenate(([0.0], np.cumsum(index * y)))
    s2 = np.concatenate(([0.0], np.cumsum(y * y)))
    line = np.full(len(y), np.nan)
    lengths = np.full(len(y), np.nan)
    length = min_length
    level = slope = spread = None
    for t in range(min_length - 1, len(y)):
        # Compare today with the forecast made from data up to yesterday.
        if level is not None:
            forecast = level + slope
            if spread > 0 and abs(y[t] - forecast) > surprise * spread:
                length = min_length
            else:
                length = min(length + 1, max_length)
        length = min(length, t + 1)
        level, slope, spread = _fit_line(s0, s1, s2, t, length)
        line[t] = level
        lengths[t] = length
    return line, lengths


def adaptive_window_alarms(y, min_length, max_length, surprise, delta):
    # Fit the changing window, then apply the shared alarm rule.
    line, _ = adaptive_window_line(y, min_length, max_length, surprise)
    return direction_flip_alarms(line, delta)


def fixed_window_alarms(y, length, delta):
    # Control version with a window that never changes.
    line, _ = adaptive_window_line(y, length, length, surprise=1e9)
    return direction_flip_alarms(line, delta)
