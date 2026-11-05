# Shared alarm rule used by every method that estimates a trend line.
# The rule follows the line and remembers its latest high and low.
# An alarm fires when the line moves against its direction by at least delta.

import numpy as np


def direction_flip_alarms(line, delta):
    # Find the first usable value.
    line = np.asarray(line, dtype=float)
    valid = np.flatnonzero(~np.isnan(line))
    alarms = []
    if len(valid) == 0:
        return alarms
    start = valid[0]
    high = low = line[start]
    direction = 0
    for t in range(start + 1, len(line)):
        value = line[t]
        if direction >= 0 and value > high:
            high = value
        if direction <= 0 and value < low:
            low = value
        # The first move only sets the direction, it is not a change.
        if direction == 0:
            if value - low >= delta:
                direction, high = 1, value
            elif high - value >= delta:
                direction, low = -1, value
        elif direction == 1 and high - value >= delta:
            alarms.append((t, -1))
            direction, low = -1, value
        elif direction == -1 and value - low >= delta:
            alarms.append((t, 1))
            direction, high = 1, value
    return alarms
