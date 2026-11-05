# Method 2, adaptive smoothing.
# Same formula as Method 1, but alpha is recalculated every day.
# The signal is the Trigg and Leach tracking signal.
# It compares the average surprise with the average size of surprises.
# Random noise cancels and gives a small signal.
# Surprises that keep the same sign give a signal near 1.
# The triangle inequality keeps the signal between 0 and 1.

import numpy as np
from src.detector import direction_flip_alarms


def adaptive_smooth(y, alpha_min, alpha_max, memory=0.1):
    # Keep the smoothed line and the alpha used each day.
    x = np.empty(len(y))
    alphas = np.empty(len(y))
    x[0] = y[0]
    alphas[0] = alpha_min
    mean_error = 0.0
    mean_size = 1e-12
    for t in range(1, len(y)):
        error = y[t] - x[t - 1]
        # Alpha for today uses only surprises seen before today.
        signal = abs(mean_error) / mean_size
        alpha = alpha_min + (alpha_max - alpha_min) * signal
        x[t] = x[t - 1] + alpha * error
        alphas[t] = alpha
        # Update the two running averages after using them.
        mean_error += memory * (error - mean_error)
        mean_size += memory * (abs(error) - mean_size)
    return x, alphas


def adaptive_smoothing_alarms(y, alpha_min, alpha_max, delta):
    # Only the smoothing differs from Method 1.
    line, _ = adaptive_smooth(y, alpha_min, alpha_max)
    return direction_flip_alarms(line, delta)
