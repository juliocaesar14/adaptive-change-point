# Step 2, test all methods on fake markets where the truth is known.

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import config
from evaluate import score, inside, bootstrap_interval
from src.simulate import CASES
from src.runner import METHOD_NAMES, tune, alarms_for, alarm_days


def make_runs(case, first_seed, count):
    # Build runs for one case. Tuning and testing use different seeds.
    runs = []
    for seed in range(first_seed, first_seed + count):
        data = CASES[case](seed, config.SIM_DAYS)
        runs.append((data["y"], data["related"], data["truth"], config.BURN_IN, config.SIM_DAYS))
    return runs


def main():
    # Tune each method on each case, then score it on fresh runs.
    rows, tests, chosen = [], [], {}
    for case_id, case in enumerate(CASES):
        tune_runs = make_runs(case, 1000 * (case_id + 1), config.SIM_TUNE_RUNS)
        test_runs = make_runs(case, 100000 * (case_id + 1), config.SIM_TEST_RUNS)
        per_method = {}
        for name in METHOD_NAMES:
            settings, _ = tune(name, tune_runs, config.TOLERANCE_DAYS)
            chosen[f"{case} | {name}"] = settings
            found = []
            for y, related, truth, low, high in test_runs:
                days = inside(alarm_days(alarms_for(name, settings, y, related)), low, high)
                found.append(score(inside(truth, low, high), days, config.TOLERANCE_DAYS, high - low))
            table = pd.DataFrame(found)
            per_method[name] = table["f1"].to_numpy()
            low_ci, high_ci = bootstrap_interval(table["f1"])
            rows.append({"case": case, "method": name, "f1": table["f1"].mean(), "f1_low": low_ci,
                         "f1_high": high_ci, "precision": table["precision"].mean(),
                         "recall": table["recall"].mean(), "mean_delay": table["mean_delay"].mean(),
                         "false_per_1000": table["false_per_1000"].mean()})
            print(case, name, round(table["f1"].mean(), 3), settings, flush=True)
        # Paired tests on the same runs answer the adaptive question directly.
        pairs = [("Adaptive smoothing", "Fixed smoothing"), ("Adaptive window", "Fixed window control"),
                 ("Adaptive smoothing", "CUSUM"), ("Adaptive window", "CUSUM"), ("Regression", "CUSUM")]
        for first, second in pairs:
            difference = per_method[first] - per_method[second]
            p_value = wilcoxon(difference).pvalue if np.any(difference != 0) else 1.0
            tests.append({"case": case, "first": first, "second": second,
                          "mean_difference": difference.mean(), "p_value": p_value})
    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "simulation_scores.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(config.RESULTS_DIR, "simulation_paired_tests.csv"), index=False)
    with open(os.path.join(config.RESULTS_DIR, "simulation_settings.json"), "w") as handle:
        json.dump(chosen, handle, indent=2)


if __name__ == "__main__":
    main()
