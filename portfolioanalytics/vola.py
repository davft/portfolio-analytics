# https://github.com/jasonstrimpel/volatility-trading/tree/master/volatility/models

import math as m
import numpy as np


def vol_garman_klass(prices, window=22, trading_periods=252, clean=True):
    # https://github.com/jasonstrimpel/volatility-trading/blob/master/volatility/models/GarmanKlass.py
    assert set(["High", "Low", "Open", "Close"]).issubset(prices.columns)

    log_hl = (prices["High"] / prices["Low"]).apply(np.log)
    log_co = (prices["Close"] / prices["Open"]).apply(np.log)

    rs = .5 * log_hl ** 2 - (2 * m.log(2) - 1) * log_co ** 2

    def f(v):
        return (trading_periods * v.mean()) ** .5

    result = rs.rolling(window=window, center=False).apply(func=f)

    if clean:
        return result.dropna()
    else:
        return result


def vol_hodges_tompkins(prices, window=2, trading_periods=252, clean=True):
    # https://github.com/jasonstrimpel/volatility-trading/blob/master/volatility/models/HodgesTompkins.py
    assert "Close" in prices.columns

    log_return = (prices["Close"] / prices["Close"].shift(1)).apply(np.log)

    vol = log_return.rolling(window=window, center=False).std() * m.sqrt(trading_periods)

    h = window
    n = (log_return.count() - h) + 1

    adj_factor = 1.0 / (1.0 - (h / n) + ((h ** 2 - 1) / (3 * n ** 2)))

    result = vol * adj_factor

    if clean:
        return result.dropna()
    else:
        return result


def vol_parkinson(prices, window=22, trading_periods=252, clean=True):
    # https://github.com/jasonstrimpel/volatility-trading/blob/master/volatility/models/Parkinson.py
    assert set(["High", "Low"]).issubset(prices.columns)

    rs = (1.0 / (4.0 * m.log(2.0))) * ((prices["High"] / prices["Low"]).apply(np.log)) ** 2.0

    def f(v):
        return (trading_periods * v.mean()) ** .5

    result = rs.rolling(window=window, center=False).apply(func=f)

    if clean:
        return result.dropna()
    else:
        return result


def vol_raw(prices, window=22, trading_periods=252, clean=True):
    # https://github.com/jasonstrimpel/volatility-trading/blob/master/volatility/models/Raw.py
    assert "Close" in prices.columns

    log_return = (prices["Close"] / prices["Close"].shift(1)).apply(np.log)

    result = log_return.rolling(window=window, center=False).std() * m.sqrt(trading_periods)

    if clean:
        return result.dropna()
    else:
        return result


def vol_rogers_satchell(prices, window=22, trading_periods=252, clean=True):
    # https://github.com/jasonstrimpel/volatility-trading/blob/master/volatility/models/RogersSatchell.py
    assert set(["High", "Low", "Open", "Close"]).issubset(prices.columns)

    log_ho = (prices["High"] / prices["Open"]).apply(np.log)
    log_lo = (prices["Low"] / prices["Open"]).apply(np.log)
    log_co = (prices["Close"] / prices["Open"]).apply(np.log)

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    def f(v):
        return (trading_periods * v.mean()) ** .5

    result = rs.rolling(window=window, center=False).apply(func=f)

    if clean:
        return result.dropna()
    else:
        return result


def vol_yang_zhang(prices, window=22, trading_periods=252, clean=True):
    # https://github.com/jasonstrimpel/volatility-trading/blob/master/volatility/models/YangZhang.py
    assert set(["High", "Low", "Open", "Close"]).issubset(prices.columns)

    log_ho = (prices["High"] / prices["Open"]).apply(np.log)
    log_lo = (prices["Low"] / prices["Open"]).apply(np.log)
    log_co = (prices["Close"] / prices["Open"]).apply(np.log)

    log_oc = (prices["Open"] / prices["Close"].shift(1)).apply(np.log)
    log_oc_sq = log_oc ** 2

    log_cc = (prices["Close"] / prices["Close"].shift(1)).apply(np.log)
    log_cc_sq = log_cc ** 2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))

    k = .34 / (1.34 + (window + 1) / (window - 1))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * m.sqrt(trading_periods)

    if clean:
        return result.dropna()
    else:
        return result


def kurtosis(prices, window=22, clean=True):
    # https://github.com/jasonstrimpel/volatility-trading/blob/master/volatility/models/Kurtosis.py
    assert "Close" in prices.columns

    log_return = (prices["Close"] / prices["Close"].shift(1)).apply(np.log)

    result = log_return.rolling(window=window, center=False).kurt()

    if clean:
        return result.dropna()
    else:
        return result


def skewness(prices, window=22, clean=True):
    # https://github.com/jasonstrimpel/volatility-trading/blob/master/volatility/models/Skew.py
    assert "Close" in prices.columns

    log_return = (prices["Close"] / prices["Close"].shift(1)).apply(np.log)

    result = log_return.rolling(window=window, center=False).skew()

    if clean:
        return result.dropna()
    else:
        return result
