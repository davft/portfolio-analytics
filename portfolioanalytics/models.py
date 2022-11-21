import pandas as pd
import warnings
from . import functions as pa


def risk_parity(ts, lookback=12, risk_budget=.1, freq=None, returns=True, max_leverage=None, rebase_to_one=False):
    """
    output: WEIGHTS
    :param ts: prices or returns timeseries. default wants returns
    :param lookback: number of periods to include in each calculation
    :param risk_budget: float, risk budget for the risk parity model
    :param freq: if None dont change the frequency of the timeseries. if str, then changes freq with change_freq_ts()
    :param returns: boolean. if False ts must be prices
    :param max_leverage: if None then dont do anything, if float then limit to max_weight
    :param rebase_to_one: if True then rebase all weights to 1
    """

    assert isinstance(ts, pd.DataFrame)

    if not returns:
        ts = pa.compute_returns(ts)

    if freq is not None:
        ts = pa.change_freq_ts(ts, freq=freq)

    # compute rolling volatility
    rollvol = pa.compute_rolling_volatility(ts, window=lookback)
    # compute weights
    weights = risk_budget / rollvol
    # drop NAs
    weights = weights.dropna()

    #
    if rebase_to_one:
        weights = weights / weights.sum(axis=1)
    elif max_leverage is not None:
        tol = 1e-06
        if any(weights.sum(axis=1) > max_leverage + tol):
            warnings.warn("\nsum of weights exceed max_leverage value of {} in dates {}:\nrebasing to {}".format(
                max_leverage, weights[weights.sum(axis=1) > max_leverage].index.values, max_leverage))
            # ribasa i pesi per le date in cui superano max_leverage + tolleranza
            weights[weights.sum(axis=1) > max_leverage] = \
                weights[weights.sum(axis=1) > max_leverage].apply(lambda x: x / sum(x) * max_leverage, axis=1)

    return weights
