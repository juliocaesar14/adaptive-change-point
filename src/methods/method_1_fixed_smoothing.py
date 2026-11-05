# Method 1, fixed smoothing.
# Formula, new guess equals old guess plus alpha times the surprise.
# Alpha never changes. This is Equation 4 in the paper.

import numpy as np
from src.detector import direction_flip_alarms


def fixed_smooth(y, alpha):
    # Build the smoothed line one day at a time.
    # Each value uses only today and the past.
    x = np.empty(len(y))
    x[0] = y[0]
    for t in range(1, len(y)):
        x[t] = x[t - 1] + alpha * (y[t] - x[t - 1])
    return x


def fixed_smoothing_alarms(y, alpha, delta):
    # Smooth first, then look for a large enough direction flip.
    return direction_flip_alarms(fixed_smooth(y, alpha), delta)
