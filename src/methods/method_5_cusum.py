# Method 5, CUSUM, the standard benchmark from Page in 1954.
# Add up standardised daily returns in two running totals.
# One total watches for upward shifts, the other for downward shifts.
# Slack ignores small returns, the limit decides when to raise an alarm.

import numpy as np
import pandas as pd


def standardised_returns(y, lookback=250):
    # Mean and spread come only from days before today.
    returns = pd.Series(np.diff(y, prepend=y[0]))
    past = returns.shift(1).rolling(lookback, min_periods=60)
    return ((returns - past.mean()) / past.std()).to_numpy()


def cusum_alarms_from_scores(scores, slack, limit):
    # Run the two totals and restart both after each alarm.
    alarms = []
    up = down = 0.0
    for t, z in enumerate(scores):
        if np.isnan(z):
            continue
        up = max(0.0, up + z - slack)
        down = max(0.0, down - z - slack)
        if up > limit:
            alarms.append((t, 1))
            up = down = 0.0
        elif down > limit:
            alarms.append((t, -1))
            up = down = 0.0
    return alarms


def cusum_alarms(y, slack, limit):
    # Join the two steps above.
    return cusum_alarms_from_scores(standardised_returns(y), slack, limit)
