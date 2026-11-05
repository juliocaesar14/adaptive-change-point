# Central settings for the whole project.

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Each data source has its own dates.
# Tuning happens before the split date, testing happens after it.
SOURCES = {
    "bundled": {"start": "2004-01-01", "end": "2022-12-31",
                "tune_from": "2005-01-01", "split": "2015-01-01"},
    "yahoo": {"start": "2004-01-01", "end": "2025-12-31",
              "tune_from": "2005-01-01", "split": "2016-01-01"},
}

TARGET_TICKER = "^GSPC"
RELATED_TICKERS = ["^IXIC", "^DJI", "^RUT"]

# Scoring rules, fixed before looking at any results.
TURN_SIZE = 0.10
TOLERANCE_DAYS = 40
BURN_IN = 250
RANDOM_DRAWS = 2000

# Simulation size.
SIM_DAYS = 1500
SIM_TUNE_RUNS = 30
SIM_TEST_RUNS = 200

# Settings tried during tuning.
DELTAS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
GRID_FIXED = {"alpha": [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]}
GRID_ADAPTIVE = {"alpha_min": [0.01, 0.02, 0.05], "alpha_max": [0.10, 0.20, 0.40]}
GRID_FIXED_WINDOW = {"length": [10, 20, 40, 60, 120]}
GRID_ADAPTIVE_WINDOW = {"min_length": [10, 20], "max_length": [40, 60, 120],
                        "surprise": [2.0, 2.5, 3.0]}
GRID_REGRESSION = {"length": [60, 120, 250], "top_k": [2, 4]}
REGRESSION_THRESHOLDS = [2.0, 2.5, 3.0, 4.0]
GRID_CUSUM = {"slack": [0.25, 0.5, 0.75, 1.0]}
CUSUM_LIMITS = [3.0, 4.0, 5.0, 7.0, 10.0]
