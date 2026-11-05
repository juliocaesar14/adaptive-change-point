# Simple checks that the maths and the fairness rules hold.
# Run this file directly with Python.

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import score, turning_points
from src.methods.method_1_fixed_smoothing import fixed_smooth
from src.methods.method_2_adaptive_smoothing import adaptive_smooth
from src.methods.method_3_adaptive_window import adaptive_window_line
from src.methods.method_4_regression import regression_gap
from src.methods.method_5_cusum import standardised_returns


def test_worked_example():
    # Same small example as the README, 10 then 11 then 11.9.
    line = fixed_smooth(np.array([10.0, 20.0, 20.0]), 0.1)
    assert np.allclose(line, [10.0, 11.0, 11.9])


def test_line_fit_matches_numpy():
    # A window of 30 must give the same end value as a standard fit.
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=100))
    line, _ = adaptive_window_line(y, 30, 30, 1e9)
    slope, intercept = np.polyfit(np.arange(30), y[70:100], 1)
    assert np.isclose(line[99], intercept + slope * 29)


def test_alpha_stays_inside_limits():
    # The triangle inequality must keep alpha between its limits.
    rng = np.random.default_rng(2)
    _, alphas = adaptive_smooth(np.cumsum(rng.standard_t(3, 3000)), 0.02, 0.3)
    assert alphas.min() >= 0.02 - 1e-12 and alphas.max() <= 0.3 + 1e-12


def test_no_method_sees_the_future():
    # Changing later days must not change any earlier output.
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0, 0.01, 600))
    related = np.cumsum(rng.normal(0, 0.01, (600, 4)), axis=0)
    y2, related2 = y.copy(), related.copy()
    y2[400:] += 5.0
    related2[400:] -= 3.0
    pairs = [
        (fixed_smooth(y, 0.1), fixed_smooth(y2, 0.1)),
        (adaptive_smooth(y, 0.02, 0.3)[0], adaptive_smooth(y2, 0.02, 0.3)[0]),
        (adaptive_window_line(y, 20, 60)[0], adaptive_window_line(y2, 20, 60)[0]),
        (regression_gap(y, related, 120, 2), regression_gap(y2, related2, 120, 2)),
        (standardised_returns(y), standardised_returns(y2)),
    ]
    for first, second in pairs:
        assert np.array_equal(first[:400], second[:400], equal_nan=True)


def test_scoring_by_hand():
    # Two true dates, one good alarm, one early alarm, one extra alarm.
    result = score(np.array([100, 200]), np.array([90, 110, 300]), 40)
    assert result["hits"] == 1
    assert np.isclose(result["precision"], 1 / 3)
    assert np.isclose(result["recall"], 1 / 2)
    assert np.isclose(result["f1"], 0.4)


def test_turning_points_by_hand():
    # Up to 1, down to 0.5, up to 1.2. The high and low must be found.
    y = np.array([0.0, 0.5, 1.0, 0.7, 0.5, 0.8, 1.2, 1.1])
    assert list(turning_points(y, 0.3)) == [2, 4]


if __name__ == "__main__":
    for name, item in list(globals().items()):
        if name.startswith("test_"):
            item()
            print("passed", name)
