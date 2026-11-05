# Step 3, tune on the early years, then test once on the later years.

import argparse
import json
import os
import numpy as np
import pandas as pd
import config
from evaluate import score, inside, turning_points, random_alarm_test
from src.runner import METHOD_NAMES, tune, alarms_for, alarm_days
from step1_prepare_data import load_log_prices


def main(source):
    # Find the positions of the tuning and testing periods.
    dates, y, related = load_log_prices(source)
    info = config.SOURCES[source]
    tune_low = int(np.searchsorted(dates, pd.Timestamp(info["tune_from"])))
    split = int(np.searchsorted(dates, pd.Timestamp(info["split"])))
    end = len(y)
    truth = turning_points(y, config.TURN_SIZE)
    rows, sensitivity, chosen, saved = [], [], {}, {}
    for name in METHOD_NAMES:
        settings, _ = tune(name, [(y, related, truth, tune_low, split)], config.TOLERANCE_DAYS)
        chosen[name] = settings
        alarms = alarms_for(name, settings, y, related)
        days = alarm_days(alarms)
        saved[name] = [{"date": str(dates[d].date()), "direction": s} for d, s in alarms if d >= tune_low]
        for period, low, high in (("tune", tune_low, split), ("test", split, end)):
            found = score(inside(truth, low, high), inside(days, low, high), config.TOLERANCE_DAYS, high - low)
            mean_random, p_value = random_alarm_test(inside(truth, low, high), low, high, found["alarms"],
                                                     found["f1"], config.TOLERANCE_DAYS, config.RANDOM_DRAWS)
            rows.append({"method": name, "period": period, **found, "random_f1": mean_random, "p_value": p_value})
        # Test scores under other answer keys and other time allowances.
        for size in (0.07, 0.10, 0.15):
            other = turning_points(y, size)
            for tolerance in (20, 40, 60):
                found = score(inside(other, split, end), inside(days, split, end), tolerance, end - split)
                sensitivity.append({"method": name, "turn_size": size, "tolerance": tolerance, "f1": found["f1"]})
        print(name, settings, flush=True)
    results = config.RESULTS_DIR
    pd.DataFrame(rows).to_csv(os.path.join(results, "sp500_scores.csv"), index=False)
    pd.DataFrame(sensitivity).to_csv(os.path.join(results, "sp500_sensitivity.csv"), index=False)
    points = pd.DataFrame({"date": dates[truth[truth >= tune_low]],
                           "log_price": y[truth[truth >= tune_low]]})
    points.to_csv(os.path.join(results, "sp500_turning_points.csv"), index=False)
    with open(os.path.join(results, "sp500_settings.json"), "w") as handle:
        json.dump(chosen, handle, indent=2)
    with open(os.path.join(results, "sp500_alarms.json"), "w") as handle:
        json.dump(saved, handle, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="bundled", choices=["bundled", "yahoo"])
    main(parser.parse_args().source)
