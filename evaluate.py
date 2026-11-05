# Separate scoring file. It judges alarms and does not create them.
# The rules are the same for every method.

import numpy as np


def turning_points(y, size):
    # Hindsight answer key, using the full series including the future.
    # A high or low counts only if price later reverses by at least size.
    # Returns the day of each confirmed high or low.
    y = np.asarray(y, dtype=float)
    points = []
    high_at = low_at = 0
    direction = 0
    for t in range(1, len(y)):
        if y[t] > y[high_at]:
            high_at = t
        if y[t] < y[low_at]:
            low_at = t
        if direction == 0:
            if y[t] - y[low_at] >= size:
                direction, high_at = 1, t
            elif y[high_at] - y[t] >= size:
                direction, low_at = -1, t
        elif direction == 1 and y[high_at] - y[t] >= size:
            points.append(high_at)
            direction, low_at = -1, t
        elif direction == -1 and y[t] - y[low_at] >= size:
            points.append(low_at)
            direction, high_at = 1, t
    return np.array(points, dtype=int)


def match_alarms(truth, alarm_days, tolerance):
    # Each true change can claim the first free alarm in its time window.
    # An alarm before the change is not credited, since it cannot be earned.
    alarm_days = np.sort(np.asarray(alarm_days, dtype=int))
    used = np.zeros(len(alarm_days), dtype=bool)
    delays = []
    for change in np.sort(truth):
        for i, day in enumerate(alarm_days):
            if not used[i] and change <= day <= change + tolerance:
                used[i] = True
                delays.append(day - change)
                break
    return delays


def score(truth, alarm_days, tolerance, n_days=None):
    # Precision, recall, F1, delay and false alarms per 1000 days.
    alarm_days = np.asarray(alarm_days, dtype=int)
    delays = match_alarms(truth, alarm_days, tolerance)
    hits = len(delays)
    precision = hits / len(alarm_days) if len(alarm_days) else 0.0
    recall = hits / len(truth) if len(truth) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    result = {"truth": int(len(truth)), "alarms": int(len(alarm_days)), "hits": hits,
              "precision": precision, "recall": recall, "f1": f1,
              "mean_delay": float(np.mean(delays)) if delays else float("nan")}
    if n_days:
        result["false_per_1000"] = 1000.0 * (len(alarm_days) - hits) / n_days
    return result


def inside(items, low, high):
    # Keep only days between low and high, including low, excluding high.
    items = np.asarray(items, dtype=int)
    return items[(items >= low) & (items < high)]


def random_alarm_test(truth, low, high, n_alarms, observed_f1, tolerance, draws, seed=0):
    # Could the same number of random alarms score as well?
    # Returns the average random F1 and a one sided p value.
    rng = np.random.default_rng(seed)
    if n_alarms == 0:
        return 0.0, 1.0
    scores = np.empty(draws)
    for i in range(draws):
        fake = rng.choice(np.arange(low, high), size=n_alarms, replace=False)
        scores[i] = score(truth, fake, tolerance)["f1"]
    p_value = (1 + np.sum(scores >= observed_f1)) / (draws + 1)
    return float(scores.mean()), float(p_value)


def bootstrap_interval(values, draws=2000, seed=0):
    # 95 percent interval for a mean, by resampling the simulation runs.
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    picks = rng.integers(0, len(values), (draws, len(values)))
    means = values[picks].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


if __name__ == "__main__":
    # Print the saved results in a readable form.
    import os
    import pandas as pd
    import config
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for title, name in (("Fake markets", "simulation_scores.csv"), ("Adaptive against fixed", "simulation_paired_tests.csv"),
                        ("S and P 500", "sp500_scores.csv")):
        print("\n" + title)
        print(pd.read_csv(os.path.join(config.RESULTS_DIR, name)).round(3).to_string(index=False))
