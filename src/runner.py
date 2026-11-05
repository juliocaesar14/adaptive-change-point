# One place that connects every method to its settings.
# This makes the comparison identical for all methods.

from itertools import product
import numpy as np
import config
from evaluate import score, inside
from src.detector import direction_flip_alarms
from src.methods.method_1_fixed_smoothing import fixed_smooth
from src.methods.method_2_adaptive_smoothing import adaptive_smooth
from src.methods.method_3_adaptive_window import adaptive_window_line
from src.methods.method_4_regression import regression_gap, gap_alarms
from src.methods.method_5_cusum import standardised_returns, cusum_alarms_from_scores

METHOD_NAMES = ["Fixed smoothing", "Adaptive smoothing", "Fixed window control",
                "Adaptive window", "Regression", "CUSUM"]


def _grid(options):
    # Turn a dictionary of lists into every combination of settings.
    keys = list(options)
    return [dict(zip(keys, values)) for values in product(*options.values())]


def all_alarm_sets(name, y, related):
    # Yield settings and alarm days for every setting of one method.
    # Slow calculations are shared across alarm limits to save time.
    if name == "Fixed smoothing":
        for s in _grid(config.GRID_FIXED):
            line = fixed_smooth(y, s["alpha"])
            for d in config.DELTAS:
                yield {**s, "delta": d}, direction_flip_alarms(line, d)
    elif name == "Adaptive smoothing":
        for s in _grid(config.GRID_ADAPTIVE):
            line, _ = adaptive_smooth(y, s["alpha_min"], s["alpha_max"])
            for d in config.DELTAS:
                yield {**s, "delta": d}, direction_flip_alarms(line, d)
    elif name == "Fixed window control":
        for s in _grid(config.GRID_FIXED_WINDOW):
            line, _ = adaptive_window_line(y, s["length"], s["length"], 1e9)
            for d in config.DELTAS:
                yield {**s, "delta": d}, direction_flip_alarms(line, d)
    elif name == "Adaptive window":
        for s in _grid(config.GRID_ADAPTIVE_WINDOW):
            line, _ = adaptive_window_line(y, s["min_length"], s["max_length"], s["surprise"])
            for d in config.DELTAS:
                yield {**s, "delta": d}, direction_flip_alarms(line, d)
    elif name == "Regression":
        for s in _grid(config.GRID_REGRESSION):
            gap = regression_gap(y, related, s["length"], s["top_k"])
            for limit in config.REGRESSION_THRESHOLDS:
                yield {**s, "threshold": limit}, gap_alarms(gap, limit)
    elif name == "CUSUM":
        scores = standardised_returns(y)
        for s in _grid(config.GRID_CUSUM):
            for limit in config.CUSUM_LIMITS:
                yield {**s, "limit": limit}, cusum_alarms_from_scores(scores, s["slack"], limit)


def alarms_for(name, settings, y, related):
    # Run one method with one chosen set of settings.
    wanted = {k: v for k, v in settings.items()}
    if name == "Fixed smoothing":
        line = fixed_smooth(y, wanted["alpha"])
        return direction_flip_alarms(line, wanted["delta"])
    if name == "Adaptive smoothing":
        line, _ = adaptive_smooth(y, wanted["alpha_min"], wanted["alpha_max"])
        return direction_flip_alarms(line, wanted["delta"])
    if name == "Fixed window control":
        line, _ = adaptive_window_line(y, wanted["length"], wanted["length"], 1e9)
        return direction_flip_alarms(line, wanted["delta"])
    if name == "Adaptive window":
        line, _ = adaptive_window_line(y, wanted["min_length"], wanted["max_length"], wanted["surprise"])
        return direction_flip_alarms(line, wanted["delta"])
    if name == "Regression":
        return gap_alarms(regression_gap(y, related, wanted["length"], wanted["top_k"]), wanted["threshold"])
    return cusum_alarms_from_scores(standardised_returns(y), wanted["slack"], wanted["limit"])


def alarm_days(alarms):
    # Keep only the day from each alarm.
    return np.array([day for day, _ in alarms], dtype=int)


def tune(name, datasets, tolerance):
    # Pick the settings with the best average F1 on the tuning data.
    # Each dataset is a tuple of y, related, truth, low and high.
    totals = {}
    for y, related, truth, low, high in datasets:
        for settings, alarms in all_alarm_sets(name, y, related):
            key = tuple(sorted(settings.items()))
            days = inside(alarm_days(alarms), low, high)
            real = inside(truth, low, high)
            totals[key] = totals.get(key, 0.0) + score(real, days, tolerance)["f1"]
    best = max(totals, key=totals.get)
    return dict(best), totals[best] / len(datasets)
