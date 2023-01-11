# https://github.com/chrisconlan/algorithmic-trading-with-python/blob/master/src/pypm/metrics.py

import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.optimize import root_scalar
import warnings
import datetime
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/16004076/python-importing-a-module-that-imports-a-module
from . import dates as dt


def rebase_ts(prices, V0=100):
    """
    rebases prices time series to V0
    :param prices: pd.Series or pd.DataFrame
    :param V0: rebase to level V0
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("`prices` must be either pd.Series or pd.DataFrame")

    if isinstance(prices, pd.Series):
        if np.isnan(prices.iloc[0]):
            # se la prima osservazione è NaN
            ts = prices.dropna()
        else:
            ts = prices

        ts_rebased = ts / ts.iloc[0] * V0

    else:
        # case 2: isinstance(prices, pd.DataFrame)
        if any(prices.iloc[0].isna()):
            # se vi è almeno un NaN nella prima osservazione delle serie storiche
            ts_rebased = list()
            for col in prices.columns:
                ts = prices[col].dropna()
                ts_rebased.append(ts / ts.iloc[0] * V0)
            ts_rebased = pd.concat(ts_rebased, axis=1)
        else:
            ts_rebased = prices / prices.iloc[0] * V0

    # nel caso in cui ci siano NaN, la serie storica in output potrebbe avere un indice diverso rispetto a prices
    ts_rebased = ts_rebased.reindex(prices.index)

    return ts_rebased


def yield_to_ts(yields, V0=100):
    """
    Computes timeseries starting from yield values. Uses ACT/360.
    :param yields:
    :param V0:
    :return:
    """
    elapsed_days = yields.index.to_series().diff().dt.days.fillna(0)
    returns = (1 + yields / 100).pow(elapsed_days / 360, axis=0) - 1
    ts = compute_ts(returns, V0=V0)

    return ts


def compute_returns(prices, method="simple"):
    """
    compute simple or log returns given a pd.Series or a pd.Dataframe
    compute returns of `price_series`.
    :param prices: pd.Series or pd.DataFrame
    :param method: "simple" or "log"
    """

    if method == "simple":
        ret_fun = lambda x: x / x.shift(1, fill_value=x[0]) - 1
    elif method == "log":
        ret_fun = lambda x: np.log(x / x.shift(1, fill_value=x[0]))
    else:
        raise ValueError("`method` must be either 'simple' or 'log'")

    if isinstance(prices, pd.Series):
        returns = ret_fun(prices)
    elif isinstance(prices, pd.DataFrame):
        returns = prices.apply(ret_fun, axis=0)
    else:
        raise TypeError("prices must be either pd.Series or pd.DataFrame")

    return returns


def compute_ts(returns, method="simple", V0=100):
    """
    compute time series given a pd.Series or a pd.Dataframe containing returns (simple or log)
    compute prices time series of `returns`.
    NB: first row of `returns` is assumed to be 0.
    :param returns: pd.Series or pd.DataFrame
    :param method: "simple" or "log"
    :param V0: int, starting value
    :return: prices time series
    """

    if method == "simple":
        ts_fun = lambda x: V0 * np.cumprod(1 + x)
    elif method == "log":
        ts_fun = lambda x: V0 * np.exp(np.cumsum(returns))
    else:
        raise ValueError("`method` must be either 'simple' or 'log'")

    if isinstance(returns, pd.Series):
        prices = ts_fun(returns)
    elif isinstance(returns, pd.DataFrame):
        prices = returns.apply(ts_fun, axis=0)
    else:
        raise TypeError("`prices` must be either pd.Series or pd.DataFrame")

    return prices


def change_freq_ts(prices, freq="monthly"):
    """
    subsets of prices timeseries with desired frequency
    :param prices: pd.Series or pd.DataFrame
    :param freq: str: weekly, monthly, yearly (or: w, m, y)
    :return: subset of prices with end of week/month/year obs
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("`prices` must be either pd.Series or pd.DataFrame")

    if not isinstance(prices.index, pd.core.indexes.datetimes.DatetimeIndex):
        raise TypeError("`prices.index` must contains dates")

    if isinstance(freq, str):
        freq = freq.lower()
        if freq not in ["weekly", "w", "monthly", "m", "yearly", "y"]:
            raise ValueError("`freq` can be: 'weekly', 'monthly', 'yearly' (or 'w', 'm', 'y')")
    else:
        raise TypeError("`freq` must be str")

    if freq in ["weekly", "w"]:
        idx = dt.get_end_of_weeks(dates=prices.index)
    elif freq in ["monthly", "m"]:
        idx = dt.get_end_of_months(dates=prices.index)
    elif freq in ["yearly", "y"]:
        idx = dt.get_end_of_years(dates=prices.index)

    # subset prices
    prices = prices.loc[idx]

    return prices


def bootstrap_ts(ret, samples, index_dates=None):
    """

    :param ret: pd.Series containing returns
    :param samples: int, tuple or np.array. if 2d np.array, then each col contains different bootstrap.
        if 1d np.array, then single bootstrap. if int, then perform bootstrap of lenght samples.
        if tuple of lenght 2, then use it as the size of np.random.choice
    :param index_dates: None or list/pd.Index of dates of length the number of extractions in each bootstrap
    :return: pd.Series or pd.DataFrame with bootstrapped returns. pd.DataFrame if samples is 2d np.array
    """

    assert isinstance(ret, pd.Series)
    if isinstance(samples, np.ndarray):
        if samples.ndim == 1:
            # 1d array
            sim_ret = ret.iloc[samples].reset_index(drop=True)
        else:
            # 2d array
            sim_ret = []
            for i in range(samples.shape[1]):
                sim_ret_i = ret.iloc[samples[:, i]].reset_index(drop=True)
                sim_ret.append(sim_ret_i)
            sim_ret = pd.concat(sim_ret, axis=1)
    elif isinstance(samples, (int, float)):
        samples = np.random.choice(range(len(ret)), size=samples)
        return bootstrap_ts(ret, samples=samples, index_dates=index_dates)
    elif isinstance(samples, tuple):
        # nel caso in cui venga passata una tupla più lunga di due, prendi solo i primi due elementi
        samples = np.random.choice(range(len(ret)), size=samples[:2])
        return bootstrap_ts(ret, samples=samples, index_dates=index_dates)
    else:
        raise Exception(f"samples must be int, tuple or np.array")

    # add zeros as first obs to compute prices time series easily later
    if isinstance(sim_ret, pd.Series):
        sim_ret = pd.concat([pd.Series(0), sim_ret])
    elif isinstance(sim_ret, pd.DataFrame):
        sim_ret = pd.concat([pd.DataFrame(0, index=[0], columns=sim_ret.columns), sim_ret], axis=0)
    else:
        print("something went wrong")

    if index_dates is not None:
        if isinstance(index_dates, (list, pd.DatetimeIndex)):
            if len(index_dates) == len(sim_ret):
                sim_ret.index = index_dates
            else:
                print(f"`index_dates` must be of lenght {len(sim_ret)}")
                print(f"Dates not added as index")
        else:
            print(f"`index_dates` must be `list` or `pd.DatetimeIndex`, got {type(index_dates)} instead")
            print(f"Dates not added as index")

    return sim_ret


def bootstrap_plot(ret, T=30, I=500, init_investm=1000, monthly_investm=None, oneoff_investm=None, seed=None):
    """

    :param ret: pd.Series with returns
    :param T: years to simulate
    :param I: number of simulations
    :param init_investm: int, initial investment
    :param monthly_investm: int, monthly investment
    :param oneoff_investm: int, one-off investment
    :return: plot
    """

    assert isinstance(ret, pd.Series), "ret must be pd.Series"
    assert isinstance(init_investm, (int, float)), "init_investm must be int"

    if seed is not None:
        np.random.seed(seed)

    # number of periods in each year
    M = 12
    # dates to simulate
    sim_dates = pd.bdate_range(ret.index.max(), "2100-12-31", freq="M")
    # select only the needed simulation dates
    sim_dates = sim_dates[:(T * M)]

    # sample length
    K = len(sim_dates) - 1

    # bootstrap
    ret_bs = bootstrap_ts(ret, samples=(K, I), index_dates=sim_dates)
    # compute time-series from bootstrapped returns
    ts_bs = compute_ts(ret_bs, V0=init_investm)

    # compute mean
    mean_ts = ts_bs.mean(axis=1)
    # compute std
    std_ts = ts_bs.std(axis=1)

    cagr = compute_cagr(mean_ts)
    all_period_ret = compute_allperiod_returns(mean_ts)
    
    tot_investm = init_investm
    final_value = mean_ts.iloc[-1]

    if oneoff_investm is not None:
        # oneoff_ts_bs = compute_ts(ret_bs, V0=oneoff_investm)
        # mean_oneoff = oneoff_ts_bs.mean(axis=1)
        # mean_oneoff = rebase_ts(mean_ts, V0=oneoff_investm)
        oneoff_ts = rebase_ts(ts_bs, V0=oneoff_investm)
        oneoff_ts = ts_bs.add(oneoff_ts, fill_value=0)
        mean_oneoff = oneoff_ts.mean(axis=1)
        std_oneoff = oneoff_ts.std(axis=1)
        cagr = compute_cagr(mean_oneoff)
        all_period_ret = compute_allperiod_returns(mean_oneoff)
        tot_investm += oneoff_investm
        final_value = mean_oneoff.iloc[-1]
    
    if monthly_investm is not None:
        for dd in sim_dates[:-1]:  # non considerare l'ultima data
            # m_ts_bs = compute_ts(ret_bs.loc[ret_bs.index >= dd], V0=monthly_investm)
            # mean_tmp = rebase_ts(mean_ts.loc[mean_ts.index >= dd], V0=monthly_investm)
            mean_tmp = rebase_ts(ts_bs.loc[mean_ts.index >= dd], V0=monthly_investm)
            try:
                monthly_ts = monthly_ts.add(mean_tmp, fill_value=0)
            except:
                monthly_ts = mean_tmp.copy()

        # mean_monthly = monthly_ts.mean(axis=1)
        monthly_ts = ts_bs.add(monthly_ts, fill_value=0)
        std_monthly = monthly_ts.std(axis=1)

        mean_monthly = monthly_ts.mean(axis=1)
        # per il calcolo del CAGR bisogna tenere in considerazione tutti gli investimenti
        final_value = mean_monthly.iloc[-1]
        cagr = compute_irr(T=T, S0=tot_investm, monthly_s=monthly_investm, F=final_value)
        # cagr = compute_cagr(mean_monthly)
        all_period_ret = (1 + cagr) ** T - 1
        tot_investm += monthly_investm * (len(sim_dates) - 1)
        

    # plot
    fig, ax = plt.subplots()
    ax.plot(mean_ts.index, mean_ts, label=f"Investment: {init_investm:,.0f}")
    if oneoff_investm is not None:
        ax.plot(mean_oneoff.index, mean_oneoff, "r", label=f"Adding one-off investment: {oneoff_investm:,.0f}")
        ax.fill_between(mean_oneoff.index, mean_oneoff - .2 * std_oneoff, mean_oneoff + .2 * std_oneoff, alpha=.5)
        ax.fill_between(mean_oneoff.index, mean_oneoff - .5 * std_oneoff, mean_oneoff + .5 * std_oneoff, alpha=.3)
        ax.fill_between(mean_oneoff.index, mean_oneoff - std_oneoff, mean_oneoff + std_oneoff, alpha=.1)
    elif monthly_investm is not None:
        ax.plot(mean_monthly.index, mean_monthly, "r", label=f"With monthly investment: {monthly_investm:,.0f}")
        ax.fill_between(mean_monthly.index, mean_monthly - .2 * std_monthly, mean_monthly + .2 * std_monthly, alpha=.5)
        ax.fill_between(mean_monthly.index, mean_monthly - .5 * std_monthly, mean_monthly + .5 * std_monthly, alpha=.3)
        ax.fill_between(mean_monthly.index, mean_monthly - std_monthly, mean_monthly + std_monthly, alpha=.1)
    else:
        ax.fill_between(mean_ts.index, mean_ts - 0.2 * std_ts, mean_ts + 0.2 * std_ts, alpha=0.5)
        ax.fill_between(mean_ts.index, mean_ts - 0.5 * std_ts, mean_ts + 0.5 * std_ts, alpha=0.3)
        ax.fill_between(mean_ts.index, mean_ts - std_ts, mean_ts + std_ts, alpha=0.1)
    plt.title(f"{ret.name} {T} years Projection.\n"
              f"CAGR: {cagr:.2%}, overall return: {all_period_ret:.2%}.\n"
              f"Total investment: {tot_investm:,.0f}, Final value: {final_value:,.0f}")
    ax.legend()
    
    return fig


def compute_excess_return(ts, bmk_ts):
    """
    compute excess return. ts and bmk_ts should have the same dates, otherwise inner join
    :param ts: pd.Series or pd.DataFrame containing returns (portfolio or generic stock)
    :param bmk_ts: pd.Series containg benchmark returns
    :return pd.Series with excess return
    """
    if not isinstance(ts, (pd.Series, pd.DataFrame)):
        print("`ts` must be pd.Series or pd.DataFrame")
        return
    if not isinstance(bmk_ts, pd.Series):
        print("`bmk_ts` must be pd.Series")
        return

    if isinstance(ts, pd.Series):
        excess_return = ts.subtract(bmk_ts.loc[ts.index])
    elif isinstance(ts, pd.DataFrame):
        excess_return = ts.apply(lambda x: x.subtract(bmk_ts.loc[ts.index]), axis=0)

    excess_return = excess_return.dropna()
    return excess_return


def compute_tracking_error_vol(ret, bmk_ret):
    """
    compute tracking error volatility wrt a benchmark timeseries
    input MUST be returns time-series
    :param ret: pd.Series or pd.DataFrame (returns)
    :param bmk_ret: pd.Series or pd.DataFrame with benchmark(s) returns
    """
    if not isinstance(ret, (pd.Series, pd.DataFrame)):
        print("`ret` must be pd.Series or pd.DataFrame")
        return
    if not isinstance(bmk_ret, (pd.Series, pd.DataFrame)):
        print("`bmk_ret` must be pd.Series or pd.DataFrame")
        return
    
    if isinstance(bmk_ret, pd.Series):
        excess_return = compute_excess_return(ts=ret, bmk_ts=bmk_ret)
        tev = compute_annualized_volatility(excess_return)
        if isinstance(ret, pd.DataFrame):
            tev.name = "tev"

    elif isinstance(bmk_ret, pd.DataFrame):
        tev = list()
        for col in bmk_ret.columns:
            excess_return = compute_excess_return(ts=ret, bmk_ts=bmk_ret[col])
            pair_tev = compute_annualized_volatility(excess_return)
            if isinstance(ret, pd.DataFrame):
                pair_tev.name = col
            else:
                pair_tev = pd.Series(pair_tev, name=col)
            tev.append(pair_tev)

        tev = pd.concat(tev, axis=1)

    return tev


def get_years_past(series):
    """
    Calculate the years past according to the index of the series for use with
    functions that require annualization
    """
    start_date = series.index[0]
    end_date = series.index[-1]

    return ((end_date - start_date).days + 1) / 365.25


def compute_cagr(prices):
    """
    Calculate compounded annual growth rate
    :param prices: pd.Series or pd.DataFrame containing prices
    """
    assert isinstance(prices, (pd.Series, pd.DataFrame))

    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    value_factor = end_price / start_price
    year_past = get_years_past(prices)

    return (value_factor ** (1 / year_past)) - 1


def compute_active_return(prices, bmk_prices):
    """
    active return = (cagr(prices) - cagr(bmk_prices))
    :param prices:
    :param bmk_prices:
    :return:
    """
    # mantieni solo le date in comune
    common_dates = set(prices.index).intersection(set(bmk_prices.index))
    # cagr
    ptf_cagr = compute_cagr(prices[prices.index.isin(common_dates)])
    bmk_cagr = compute_cagr(bmk_prices[bmk_prices.index.isin(common_dates)])

    if isinstance(bmk_prices, pd.DataFrame):
        output = pd.DataFrame(np.zeros((len(ptf_cagr), len(bmk_prices.columns))), 
                              index=ptf_cagr.index, columns=bmk_prices.columns)
        for col in bmk_prices.columns:
            output[col] = ptf_cagr - bmk_cagr[col]
    else:
        output = ptf_cagr - bmk_cagr

    return output


def compute_allperiod_returns(ts, method="simple"):
    """
    Compute NON-annualized return between first and last available obs
    :param ts: pd.Series or pd.DataFrame containing prices timeseries
    :param method: str, "simple" or "log"
    :return: 
    """
    assert isinstance(ts, (pd.Series, pd.DataFrame))

    if method == "simple":
        return ts.iloc[-1] / ts.iloc[0] - 1
    elif method == "log":
        return np.log(ts.iloc[-1] / ts.iloc[0])
    else:
        raise ValueError("method should be either simple or log")
    

def compute_annualized_volatility(returns):
    """
    Calculates annualized volatility for a date-indexed return series.
    Works for any interval of date-indexed prices and returns.
    """
    years_past = get_years_past(returns)
    entries_per_year = returns.shape[0] / years_past
    return returns.std() * np.sqrt(entries_per_year)


def compute_rolling_volatility(returns, window=252):
    """
    Wrapper for compute_annualized_volatility. it computes rolling annualized volatility
    :param returns:
    :param window:
    :return:
    """
    roll_vol = returns.rolling(window=window).apply(lambda x: compute_annualized_volatility(x))
    return roll_vol


def compute_sharpe_ratio(returns, benchmark_rate=0):
    """
    Calculates the sharpe ratio given a price series. Defaults to benchmark_rate
    of zero.
    """
    prices = compute_ts(returns)
    cagr = compute_cagr(prices)
    # returns = compute_returns(prices)
    volatility = compute_annualized_volatility(returns)
    return (cagr - benchmark_rate) / volatility


def compute_information_ratio(returns, bmk_returns):
    """
    Compute the Information ratio wrt a benchmark
    IR = (ptf_returns - bmk_returns) / tracking_error_vol
    :param returns: portfolio returns. pd.Series or pd.DataFrame
    :param bmk_returns: benchmark returns. pd.Series or pd.DataFrame
    :return:
    """
    prices = compute_ts(returns)
    bmk_prices = compute_ts(bmk_returns)
    # compute active return
    active_ret = compute_active_return(prices, bmk_prices)
    # compute TEV
    tev = compute_tracking_error_vol(returns, bmk_returns)

    return active_ret / tev


def compute_rolling_sharpe_ratio(prices, window=20, method="simple"):
    """
    Compute an *approximation* of the sharpe ratio on a rolling basis.
    Intended for use as a preference value.
    """
    rolling_return_series = compute_returns(prices, method=method).rolling(window)
    return rolling_return_series.mean() / rolling_return_series.std()


def compute_beta(ret, bmk_ret, use_linear_reg=False):
    """
    Computes single-regression beta.    ret = alpha + beta * bmk_ret
    If ret is pd.Series or pd.DataFrame with 1 column:
        If bmk_ret is pd.Series or pd.DataFrame with 1 column:
            output: scalar
        Elif bmk_ret is pd.DataFrame with >1 column:
            output: pd.DataFrame with 1 row (ret.name) and len(bmk_ret.columns) cols
    Elif ret is pd.DataFrame with >1 columns:
        If bmk_ret is pd.Series or pd.DataFrame with 1 column:
            output: pd.DataFrame with 1 col (bmk_ret.name) and len(ret.columns) rows
        Elif bmk_ret is pd.DataFrame with >1 column:
            output: pd.DataFrame with len(ret.columns) rows and len(bmk_ret.columns) cols
    https://corporatefinanceinstitute.com/resources/knowledge/finance/beta-coefficient/
    :param ret: stock/portfolio returns. works with multiple timeseries as well
    :param bmk_ret: benchmark returns. works with multiple benckmarks
    :param use_linear_reg: boolean -> compute beta using linear regression?
    :return:
    """

    if not isinstance(ret, (pd.Series, pd.DataFrame)):
        raise TypeError("`ret` must be pd.Series or pd.DataFrame containing returns")
    if not isinstance(bmk_ret, (pd.Series, pd.DataFrame)):
        raise TypeError("`bmk_ret` must be pd.Series or pd.DataFrame containing benchmark returns")

    if isinstance(ret, pd.Series):
        ret = ret.to_frame(ret.name)

    if isinstance(bmk_ret, pd.Series):
        bmk_series = True
        bmk_ret = bmk_ret.to_frame(bmk_ret.name)
    else:
        bmk_series = False

    # bisogna fare in modo che le date (.index) coincidano
    if len(set(ret.index).symmetric_difference(bmk_ret.index)):
        # merge per allineare le date
        tmp = pd.concat([ret, bmk_ret], axis=1)
        # inserisci zero al posto degli NA
        # tmp = tmp.fillna(0)
        # rimuovi gli NA
        tmp = tmp.dropna()
        ret = tmp[ret.columns]
        bmk_ret = tmp[bmk_ret.columns]

    # inizializza un pd.DataFrame dove verranno salvati i beta
    beta = np.zeros((len(ret.columns), len(bmk_ret.columns)))
    beta = pd.DataFrame(beta, index=ret.columns, columns=bmk_ret.columns)

    if use_linear_reg:
        # use linear regression to compute beta
        m = LinearRegression(fit_intercept=True)

        for icol in range(len(ret.columns)):
            for jcol in range(len(bmk_ret.columns)):
                # reg: ret = alpha + beta * bmk_ret
                # bmk_ret è il ritorno del mercato nel CAPM
                m.fit(bmk_ret.iloc[:, [jcol]], ret.iloc[:, [icol]])
                beta.iloc[icol, jcol] = m.coef_[0][0]

    else:
        # use variance - covariance
        # variance and covariance
        for jcol in bmk_ret.columns:
            # serve lavorare su un unico pd.DataFrame
            tmp = pd.concat([ret, bmk_ret[[jcol]]], axis=1)
            # inserisci zero al posto degli NA
            # tmp = tmp.fillna(0)
            # rimuovi gli NA
            tmp = tmp.dropna()
            # calcola beta
            m = np.cov(tmp, rowvar=False)
            variance = m[(m.shape[0] - 1), (m.shape[1] - 1)]
            # non considerare la covarianza con il benchmark
            covariance = m[:-1, (m.shape[1] - 1)]
            # salva beta
            beta[jcol] = covariance / variance

    if bmk_series:
        # output pd.Series, not pd.DataFrame
        beta = beta[bmk_ret.columns[0]]

    if sum(beta.shape) == 1:
        # ritorna scalare
        beta = beta[0]
    elif sum(beta.shape) == 2 and not bmk_series:
        # ritorna scalare
        beta = beta.iloc[0, 0]

    return beta


def compute_multiple_beta(y, xs):
    """
    Computes multiple linear regression y = a + b_1 * xs_1 + b_2 * xs_2 + .. + b_n * xs_n
    :param y: stock/portfolio returns. must be pd.Series or pd.DataFrame with one column
    :param xs: benchmark returns. must be pd.Series or pd.DataFrame
    :return:
    """

    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("`y` must be pd.Series or pd.DataFrame containing independent variable returns")
    if not isinstance(xs, (pd.Series, pd.DataFrame)):
        raise TypeError("`xs` must be pd.Series or pd.DataFrame containing dependent variables returns")

    if isinstance(y, pd.Series):
        y = y.to_frame(y.name)
    elif isinstance(y, pd.DataFrame):
        if y.shape[1] > 1:
            print("`y` must contains only one column. subsetting `y` to obtain a 1col df")

    if isinstance(xs, pd.Series):
        xs = xs.to_frame(xs.name)

    # bisogna fare in modo che le date (.index) coincidano
    if len(set(y.index).symmetric_difference(xs.index)):
        # merge per allineare le date
        tmp = pd.concat([y, xs], axis=1)
        # inserisci zero al posto degli NA
        tmp = tmp.fillna(0)
        y = tmp[y.columns]
        xs = tmp[xs.columns]

    # inizializza un pd.DataFrame dove verranno salvati i beta
    beta = np.zeros((len(xs.columns), len(y.columns)))
    beta = pd.DataFrame(beta, index=xs.columns, columns=y.columns)

    # multiple regression
    m = LinearRegression(fit_intercept=True)
    m.fit(xs, y)
    beta[y.columns[0]] = m.coef_.transpose()

    return beta


def compute_rolling_beta(ret, bmk_ret, window=252):
    """
    Computes single-regression beta over a rolling window.    ret = alpha + beta * bmk_ret
    Works with:
        (ret is pd.Series or 1 col pd.DataFrame) and (bmk_ret is pd.Series or 1 col pd.DataFrame)
        ret is pd.DataFrame and (bmk_ret is pd.Series or 1 col pd.DataFrame)
        (ret is pd.Series or 1 col pd.DataFrame) and bmk_ret is pd.DataFrame
    https://corporatefinanceinstitute.com/resources/knowledge/finance/beta-coefficient/
    :param ret: stock/portfolio returns. works with multiple timeseries as well
    :param bmk_ret: benchmark returns. works with multiple benckmarks
    :param window: int, lenght of rolling window
    :return:
    """

    if not isinstance(ret, (pd.Series, pd.DataFrame)):
        raise TypeError("`ret` must be pd.Series or pd.DataFrame containing returns")
    if not isinstance(bmk_ret, (pd.Series, pd.DataFrame)):
        raise TypeError("`bmk_ret` must be pd.Series or pd.DataFrame containing benchmark returns")

    if isinstance(ret, pd.DataFrame) and isinstance(bmk_ret, pd.DataFrame):
        if len(ret.columns) > 1 and len(bmk_ret.columns) > 1:
            raise NotImplementedError("Function cannot process results with `ret` and `bmk_ret` with >1 cols each")

    # inizializza beta come lista
    beta = list()

    for i in range(window, ret.shape[0]):
        y = ret.iloc[(i - window):i]
        x = bmk_ret.iloc[(i - window):i]
        b = compute_beta(y, x)
        if isinstance(b, float):
            # case y is pd.Series and x is pd.Series
            b = pd.Series(b, index=[y.index.max()])
        elif isinstance(b, pd.Series):
            # case y is pd.DataFrame and x is pd.Series
            b = b.to_frame().transpose()
            b.index = [y.index.max()]
        elif isinstance(b, pd.DataFrame):
            # case y is pd.Series and x is pd.Dataframe
            b.index = [y.index.max()]

        beta.append(b)

    beta = pd.concat(beta, axis=0)

    return beta


def compute_annualized_downside_deviation(returns, benchmark_rate=0):
    """
    Calculates the downside deviation for use in the sortino ratio.
    Benchmark rate is assumed to be annualized. It will be adjusted according
    to the number of periods per year seen in the data.
    """

    # For both de-annualizing the benchmark rate and annualizing result
    years_past = get_years_past(returns)
    entries_per_year = returns.shape[0] / years_past

    adjusted_benchmark_rate = ((1 + benchmark_rate) ** (1 / entries_per_year)) - 1

    downside_series = adjusted_benchmark_rate - returns
    downside_sum_of_squares = (downside_series[downside_series > 0] ** 2).sum()
    denominator = returns.shape[0] - 1
    downside_deviation = np.sqrt(downside_sum_of_squares / denominator)

    return downside_deviation * np.sqrt(entries_per_year)


def compute_rolling_downside_deviation(returns, window=252, benchmark_rate=0):
    """
    Wrapper for compute_annualized_downside_deviation. it computes rolling annualized volatility
    :param returns:
    :param window:
    :param benchmark_rate:
    :return:
    """
    roll_dwndev = returns.rolling(window=window)\
        .apply(lambda x: compute_annualized_downside_deviation(x, benchmark_rate=benchmark_rate))
    return roll_dwndev


def compute_sortino_ratio(prices, benchmark_rate=0):
    """
    Calculates the sortino ratio.
    """
    cagr = compute_cagr(prices)
    return_series = compute_returns(prices)
    downside_deviation = compute_annualized_downside_deviation(return_series, benchmark_rate)
    return (cagr - benchmark_rate) / downside_deviation


def compute_jensens_alpha(returns, benchmark_return_series):
    """
    Calculates jensens alpha. Prefers input series have the same index. Handles
    NAs.
    :param returns: pd.Series
    :param benchmark_return_series: pd.Series with benchmark return
    """

    if isinstance(returns, pd.DataFrame):
        warnings.warn("returns must be pd.Series, taking the first column: {}".format(returns.columns[0]))
        returns = returns[returns.columns[0]]
    # Join series along date index and purge NAs
    df = pd.concat([returns, benchmark_return_series], sort=True, axis=1)
    df = df.dropna()

    # Get the appropriate data structure for scikit learn
    clean_returns = df[df.columns.values[0]]
    clean_benchmarks = pd.DataFrame(df[df.columns.values[1]])

    # Fit a linear regression and return the alpha
    regression = LinearRegression().fit(clean_benchmarks, y=clean_returns)
    return regression.intercept_


def compute_drawdown_series(prices, method="simple"):
    """
    Returns the drawdown series
    """

    if method == "simple":
        evaluator = lambda price, peak: (price / peak) - 1
    elif method == "log":
        evaluator = lambda price, peak: np.log(prices) - np.log(peak)
    else:
        ValueError("method should be either simple or log")

    return evaluator(prices, prices.cummax())


def compute_max_drawdown(prices, method="simple"):
    """
    Simply returns the max drawdown as a float
    """
    return compute_drawdown_series(prices, method=method).min()


def drawdown_analysis(prices, method="simple"):
    """

    :param prices: prices timeseries
    :param method:
    """
    assert isinstance(prices, pd.DataFrame), "pd.Series not implemented yet !!!"
    # compute drawdown
    drawdown = compute_drawdown_series(prices, method=method)

    def dd_single_ts(dd):
        max_dd = 0
        output = list()
        for day in dd.index:
            if dd.loc[day].values[0] == 0:
                if max_dd == 0:
                    continue
                # new peak
                tmp = pd.DataFrame([[dd_date, max_dd, day]], columns=["Drawdown Date", "Drawdown", "Peak Date"])
                output.append(tmp)
                # reset max drawdown for next iteration
                max_dd = 0
            elif dd.loc[day].values[0] < max_dd:
                dd_date = day
                max_dd = dd.loc[day].values[0]
        output = pd.concat(output, axis=0)
        # Drawdown ranking
        output["Rank"] = output["Drawdown"].rank()
        # Drawdown time to recovery
        output["Time to Recovery (Days)"] = (output["Peak Date"] - output["Drawdown Date"])
        output["Time to Recovery (Days)"] = output["Time to Recovery (Days)"].astype("timedelta64[D]")

        return output

    if len(drawdown.columns) == 1:
        analysis = dd_single_ts(drawdown)
    else:
        analysis = dict()
        for col in drawdown.columns:
            col_analysis = dd_single_ts(drawdown[[col]])
            analysis[col] = col_analysis

    return analysis


def compute_calmar_ratio(prices, years_past=None, method="simple"):
    """
    Return the percent max drawdown ratio over the past n years, otherwise
    known as the Calmar Ratio (3yr). if years_past is None, then consider all period.
    """

    if isinstance(years_past, int):
        # Filter series on past three years
        last_date = prices.index[-1]
        n_years_ago = last_date - pd.Timedelta(days=years_past * 365.25)
        prices = prices[prices.index > n_years_ago]

    # Compute annualized percent max drawdown ratio
    percent_drawdown = -compute_max_drawdown(prices, method=method)
    cagr = compute_cagr(prices)
    return cagr / percent_drawdown


def compute_parametric_var(returns, conf_lvl=.95):
    """
    :param returns pd.Series or pd.DataFrame
    :param conf_lvl: float, confidence level
    ref: https://blog.quantinsti.com/calculating-value-at-risk-in-excel-python/
    """
    mu = np.mean(returns, axis=0)
    sigma = np.std(returns, axis=0)

    # Percent Point Function
    VaR = norm.ppf(1 - conf_lvl, mu, sigma)
    
    if isinstance(returns, pd.DataFrame):
        VaR = pd.Series(VaR, index=returns.columns, name="Parametric VaR")
    
    return VaR


def compute_historical_var(returns, conf_lvl=.95):
    """
    :param returns pd.Series or pd.DataFrame
    :param conf_lvl: float, confidence level
    ref: https://blog.quantinsti.com/calculating-value-at-risk-in-excel-python/
    """
    VaR = returns.quantile(1 - conf_lvl)
    if isinstance(returns, pd.DataFrame):
        VaR.name = "Historical VaR"

    return VaR


def compute_corr(ts, df=None):
    """
    computes correlation over all given period
    if `ts` is pd.Series, then it computes `ts` vs all `df` cols
    if `ts` is pd.DataFrame and `df` is None, then it computes all pairwise corr between `ts` cols
    if `ts` is pd.DataFrame and `df` is pd.Series, then it computes all `ts` vs `df`
    :param ts: pd.Series or pd.DataFrame (returns!!)
        if pd.Series, then `df` is mandatory and must be pd.Series or pd.DataFrame
        if pd.DataFrame and `df` is not given, then all pairwise correlations are computed
        if pd.DataFrame and `df` is pd.Series, then all correlation btw `ts` and `df` are computed
    :param df: None or pd.Series, pd.DataFrame
    """

    if isinstance(ts, pd.Series) and df is None:
        print("please provide argument `df` (can't be None if `ts` is pd.Series)")
        return

    if isinstance(ts, pd.Series):
        if isinstance(df, pd.Series):
            # output: float
            corr = ts.corr(df)
            # corr.columns = [ts.name + "-" + df.name]
        elif isinstance(df, pd.DataFrame):
            # output: pd.Series
            corr = df.apply(lambda x: x.corr(ts), axis=0).reset_index(name="corr")
            # corr.columns = [ts.name + "-" + col for col in corr.columns]

    if isinstance(ts, pd.DataFrame):
        if df is None:
            # corr = pd.DataFrame(0, index=ts.columns, columns=ts.columns)
            # for i in range(len(ts.columns)):
            #     for j in range(i, len(ts.columns)):
            #         corr.iloc[i, j] = ts[ts.columns[i]].corr(ts[ts.columns[j]])
            #         # it's the same
            #         corr.iloc[j, i] = corr.iloc[i, j]
            corr = ts.corr()

        elif isinstance(df, pd.Series):
            corr = ts.apply(lambda x: x.corr(df), axis=0)
            corr.name = "corr"

        elif isinstance(df, pd.DataFrame):
            corr = list()
            for i in range(len(df.columns)):
                pair_corr = ts.apply(lambda x: x.corr(df[df.columns[i]]), axis=0)  # .reset_index(name=df.columns[i])
                pair_corr.name = df.columns[i]
                corr.append(pair_corr)
            corr = pd.concat(corr, axis=1)

    return corr


def compute_rolling_corr(ts, df=None, window=252):
    """
    compute rolling correlations. if `ts` is pd.Series, then it computes `ts` vs all `df` cols
    if `ts` is pd.DataFrame and `df` is None, then it computes all pairwise corr between `ts` cols
    if `ts` is pd.DataFrame and `df` is pd.Series, then it computes all `ts` vs `df`
    :param ts: pd.Series or pd.DataFrame
        if pd.Series, then `df` is mandatory and must be pd.Series or pd.DataFrame
        if pd.DataFrame and `df` is not given, then all pairwise correlations are computed
        if pd.DataFrame and `df` is pd.Series, then all correlation btw `ts` and `df` are computed
    :param df: None or pd.Series, pd.DataFrame
    :param window: int, rolling window
    """

    if isinstance(ts, pd.Series) and df is None:
        raise TypeError("please provide argument `df` (can't be None if `ts` is pd.Series)")

    if isinstance(ts, pd.DataFrame) and len(ts.columns) == 1:
        # if ts is pd.DataFrame with 1 col, convert to pd.Series
        ts = ts[ts.columns[0]]

    if isinstance(df, pd.DataFrame) and len(df.columns) == 1:
        # if df is pd.DataFrame with 1 col, convert to pd.Series
        df = df[df.columns[0]]

    if isinstance(ts, pd.Series):
        if isinstance(df, pd.Series):
            corr = ts.rolling(window).corr(df).to_frame()
            corr.columns = [ts.name + "-" + df.name]
        elif isinstance(df, pd.DataFrame):
            corr = df.rolling(window).apply(lambda x: x.corr(ts))
            corr.columns = [ts.name + "-" + col for col in corr.columns]

    if isinstance(ts, pd.DataFrame):
        if df is None:
            corr = list()
            for i in range(len(ts.columns) - 1):
                for j in range(i + 1, len(ts.columns)):
                    pair_corr = ts[ts.columns[i]].rolling(window).corr(ts[ts.columns[j]]).to_frame()
                    pair_corr.columns = [ts.columns[i] + "-" + ts.columns[j]]
                    corr.append(pair_corr)

            corr = pd.concat(corr, axis=1)
        elif isinstance(df, pd.Series):
            corr = list()
            for i in range(len(ts.columns)):
                pair_corr = ts[ts.columns[i]].rolling(window).corr(df).to_frame()
                pair_corr.columns = [ts.columns[i] + "-" + df.name]
                corr.append(pair_corr)

            corr = pd.concat(corr, axis=1)

    corr = corr.dropna()
    return corr


def compute_spread_ts(ts1, ts2, rebase=True):
    """
    Computes spread difference between two prices time-series
    :param ts1: price time-series
    :param ts2:
    :return:
    """
    if rebase:
        ts1 = rebase_ts(ts1)
        ts2 = rebase_ts(ts2)

    out = ts1 - ts2
    return out


def compute_ratio_ts(ts1, ts2, rebase=True):
    """
    Computes ratio between two prices time-series
    :param ts1: price time-series
    :param ts2:
    :return:
    """
    if rebase:
        ts1 = rebase_ts(ts1)
        ts2 = rebase_ts(ts2)

    out = ts1 / ts2
    return out


def compute_summary_statistics(ts, return_annualized=True):
    """
    Computes statistics (annualised return and annualised std) for various time-frame
    Time-frames are: YTD, 1m, 3m, 6m, 1y, 2y, 3y, 5y, ALL
    :param ts: pd.DataFrame with prices time series
    :param return_annualized: boolean, if True return annualized returns, otherwise periodic returns
    :return: pd.DataFrame(s) with summary statistics with column
        if ts is pd.DataFrame: | ShareClassISIN | metric | interval | value |
        if ts is pd.Series: | metric | interval | value |
    """
    assert(isinstance(ts, (pd.Series, pd.DataFrame)))
    
    # quanti anni è lunga la serie storica? in base a questo capiamo fino a che periodo possiamo calcolare
    n_years = get_years_past(ts)

    # calcola ritorni sui vari time-frame.
    last_date = ts.index.to_list()[-1]
    ytd_date = last_date + pd.tseries.offsets.YearEnd(-1)
    one_mo_date = last_date + pd.tseries.offsets.DateOffset(months=-1)
    three_mo_date = last_date + pd.tseries.offsets.DateOffset(months=-3)
    six_mo_date = last_date + pd.tseries.offsets.DateOffset(months=-6)
    one_ye_date = last_date + pd.tseries.offsets.DateOffset(years=-1)
    two_ye_date = last_date + pd.tseries.offsets.DateOffset(years=-2)
    three_ye_date = last_date + pd.tseries.offsets.DateOffset(years=-3)
    five_ye_date = last_date + pd.tseries.offsets.DateOffset(years=-5)

    ret = compute_returns(ts)
    
    if return_annualized:
        return_function = compute_cagr
        ret_type = "Annualized Return"
    else:
        return_function = compute_allperiod_returns
        ret_type = "Return"
    
    if isinstance(ts, pd.Series):
        statistics = [
            [ret_type, "YTD", return_function(ts.loc[ytd_date:])],
            [ret_type, "1 Month", return_function(ts.loc[one_mo_date:])],
            [ret_type, "3 Months", return_function(ts.loc[three_mo_date:])],
            [ret_type, "6 Months", return_function(ts.loc[six_mo_date:])],
            [ret_type, "1 Year", return_function(ts.loc[one_ye_date:])],
            [ret_type, "2 Years", return_function(ts.loc[two_ye_date:])],
            [ret_type, "3 Years", return_function(ts.loc[three_ye_date:])],
            [ret_type, "5 Years", return_function(ts.loc[five_ye_date:])],
            [ret_type, "All Period", return_function(ts)],
            ["Standard Deviation", "YTD", compute_annualized_volatility(ret.loc[ytd_date:])],
            ["Standard Deviation", "1 Month", compute_annualized_volatility(ret.loc[one_mo_date:])],
            ["Standard Deviation", "3 Months", compute_annualized_volatility(ret.loc[three_mo_date:])],
            ["Standard Deviation", "6 Months", compute_annualized_volatility(ret.loc[six_mo_date:])],
            ["Standard Deviation", "1 Year", compute_annualized_volatility(ret.loc[one_ye_date:])],
            ["Standard Deviation", "2 Years", compute_annualized_volatility(ret.loc[two_ye_date:])],
            ["Standard Deviation", "3 Years", compute_annualized_volatility(ret.loc[three_ye_date:])],
            ["Standard Deviation", "5 Years", compute_annualized_volatility(ret.loc[five_ye_date:])],
            ["Standard Deviation", "All Period", compute_annualized_volatility(ret)]
        ]
        # crea pd.DataFrame()
        statistics = pd.DataFrame(statistics, columns=["metric", "interval", "value"])
    
    else:
        # ts è pd.DataFrame()
        statistics = pd.concat([
            pd.DataFrame({"metric": ret_type, "interval": "YTD", "value": return_function(ts.loc[ytd_date:])}),
            pd.DataFrame({"metric": ret_type, "interval": "1 Month", "value": return_function(ts.loc[one_mo_date:])}),
            pd.DataFrame({"metric": ret_type, "interval": "3 Months", "value": return_function(ts.loc[three_mo_date:])}),
            pd.DataFrame({"metric": ret_type, "interval": "6 Months", "value": return_function(ts.loc[six_mo_date:])}),
            pd.DataFrame({"metric": ret_type, "interval": "1 Year", "value": return_function(ts.loc[one_ye_date:])}),
            pd.DataFrame({"metric": ret_type, "interval": "2 Years", "value": return_function(ts.loc[two_ye_date:])}),
            pd.DataFrame({"metric": ret_type, "interval": "3 Years", "value": return_function(ts.loc[three_ye_date:])}),
            pd.DataFrame({"metric": ret_type, "interval": "5 Years", "value": return_function(ts.loc[five_ye_date:])}),
            pd.DataFrame({"metric": ret_type, "interval": "All Period", "value": return_function(ts)}),
            pd.DataFrame({"metric": "Standard Deviation", "interval": "YTD", 
                          "value": compute_annualized_volatility(ret.loc[ytd_date:])}),
            pd.DataFrame({"metric": "Standard Deviation", "interval": "1 Month",
                          "value": compute_annualized_volatility(ret.loc[one_mo_date:])}),
            pd.DataFrame({"metric": "Standard Deviation", "interval": "3 Months",
                          "value": compute_annualized_volatility(ret.loc[three_mo_date:])}),
            pd.DataFrame({"metric": "Standard Deviation", "interval": "6 Months",
                          "value": compute_annualized_volatility(ret.loc[six_mo_date:])}),
            pd.DataFrame({"metric": "Standard Deviation", "interval": "1 Year",
                          "value": compute_annualized_volatility(ret.loc[one_ye_date:])}),
            pd.DataFrame({"metric": "Standard Deviation", "interval": "2 Years",
                          "value": compute_annualized_volatility(ret.loc[two_ye_date:])}),
            pd.DataFrame({"metric": "Standard Deviation", "interval": "3 Years",
                          "value": compute_annualized_volatility(ret.loc[three_ye_date:])}),
            pd.DataFrame({"metric": "Standard Deviation", "interval": "5 Years",
                          "value": compute_annualized_volatility(ret.loc[five_ye_date:])}),
            pd.DataFrame({"metric": "Standard Deviation", "interval": "All Period",
                          "value": compute_annualized_volatility(ret)})
        ])
        statistics = statistics.reset_index()

    return statistics


def compute_turnover(hldg, index="Date", columns="ISIN", values="w_rebase"):
    """
    NB: per il turnover preciso, usare Portfolio().portfolio_returns(verbose=True). quella funzione calcola
    il turnover usando i pesi end of period.
    Se si vuole calcolare il turnover di indici o ptf per cui si hanno solo i pesi puntuali alle rebalancing, ma non
    i prezzi per poter calcolare i pesi end of period, allora usare questa funzione. è un'approssimazione molto fedele.
    :param hldg: pd.DataFrame with holdings over time
    :param index: column with dates. if pd.DF is indexed by Dates, then use index=True
    :param columns: column with instruments identifier (es: "ISIN", "Ticker")
    :param values: column with weights
    :return: portfolio turnover over time
    """

    if index is True:
        # https://docs.quantifiedcode.com/python-anti-patterns/readability/comparison_to_true.html
        # Dates are in hdlg.index
        hldg = hldg.reset_index()
        index = "index"

    if not all([x in hldg.columns for x in [index, columns, values]]):
        raise IndexError(f"hldg must contains columns: {index}, {columns}, {values}")

    # pivot per calcolare più velocemente il turnover
    hh = hldg.pivot(index=index, columns=columns, values=values)
    hh = hh.fillna(0)
    
    turnover_i = list()
    for i in range(len(hh)):
        if i == 0:
            continue
        # secondo la definizione di cui sopra, il massimo turnover è 2. se modifichiamo l'allocazione dell'intero
        # ptf vogliamo turnover=1
        # turnover_i.append(np.sum(np.abs(hh.iloc[i] - hh.iloc[i - 1])) / 2)
        turnover_i.append(np.sum(np.abs(hh.iloc[i] - hh.iloc[i - 1])))
    
    turnover = pd.Series(turnover_i, index=hh.index[1:])

    return turnover


def compute_VaR(returns, formula="Parametric Normal", conf_int=.95, period_int=None,
                ewma_discount_f=.94, series=False, removeNA=True):
    """
    https://github.com/BSIC/VaR/blob/master/VaR.py
    this function can calculate both single value VaR and series of VaR values through time
    supported formulas: Parametric Normal, Parametric EWMA, Historical Simulation, Filtered Historical Simulation
    """
    # returns must be pd.Series
    if isinstance(returns, pd.DataFrame):
        # prendi solo la prima colonna
        returns = returns[returns.columns[0]]

    # removes NAs from the series
    if removeNA:
        returns = returns[pd.notnull(returns)]

    if series and period_int is None:
        # default value
        period_int = 100
    elif period_int is None:
        period_int = len(returns)

    #########################
    # Parametric Normal VaR #
    #########################
    if formula in ["Parametric Normal", "Normal"]:
        if not series:
            data = returns[-period_int:]
            stdev = np.std(data)
            VaR = stdev * norm.ppf(conf_int)
        else:
            VaR = pd.Series(index=returns.index, name="ParVaR")
            for i in range(len(returns) - period_int):
                if i == 0:
                    data = returns[-period_int:]
                else:
                    data = returns[-period_int - i:-i]
                stdev = np.std(data)
                VaR[-i - 1] = stdev * norm.ppf(conf_int)

    #######################
    # EWMA Parametric VaR #
    #######################
    if formula in ["Parametric EWMA", "EWMA"]:
        # define exponentially smoothed weights components
        dof = np.empty([period_int, ])       # degree of freedom
        weights = np.empty([period_int, ])
        dof[0] = 1.
        dof[1] = ewma_discount_f
        Range = range(period_int)
        for i in range(2, period_int):
            dof[i] = dof[1] ** Range[i]
        for i in range(period_int):
            weights[i] = dof[i] / sum(dof)

        if not series:
            sqrd_data = returns[-period_int:] ** 2
            ewma_stdev = math.sqrt(sum(weights * sqrd_data))
            VaR = ewma_stdev * norm.ppf(conf_int)
        else:
            VaR = pd.Series(index=returns.index, name="EWMAVaR")
            sqrd_returns = returns ** 2
            # this loop repeas the VaR calculation iterated for every xxx period interval
            for i in range(len(returns) - period_int):
                # this is needed as, supposing x is a number, referencing a pd.series as a[x, 0] is a mistake.
                # correct is a[x:]
                if i == 0:
                    sqrd_data = sqrd_returns[-period_int:]
                else:
                    sqrd_data = sqrd_returns[-period_int - i:-i]
                ewma_stdev = math.sqrt(sum(weights * sqrd_data))
                # pd.series work differently for singular entries. so if a[x:] gets up to the last number, a[] does not
                # work. so a[-1] will get the equivalent to the last of a[x:-1]
                VaR[-i - 1] = ewma_stdev * norm.ppf(conf_int)

    #########################
    # Historical Simulation #
    #########################
    if formula in ["Historical Simulation", "Historical"]:
        if not series:
            data = returns[-period_int:]
            VaR = -np.percentile(data, 1 - conf_int)
        else:
            VaR = pd.Series(index=returns.index, name="HSVaR")
            for i in range(len(returns) - period_int):
                if i == 0:
                    data = returns[-period_int:]
                else:
                    data = returns[-period_int - i:-i]
                VaR[-i - 1] = -np.percentile(data, 1 - conf_int)

    ##################################
    # Filtered Historical Simulation #
    ##################################
    if formula in ["Filtered Historical Simulation", "Filtered", "FHS"]:
        # defining exponentially smoothed weights components
        dof = np.empty([period_int, ])
        weights = np.empty([period_int, ])
        dof[0] = 1.
        dof[1] = ewma_discount_f
        Range = range(period_int)
        for i in range(2, period_int):
            dof[i] = dof[1] ** Range[i]
        for i in range(period_int):
            weights[i] = dof[i] / sum(dof)

        VaR = pd.Series(index=returns.index, name="FHSVaR")
        ewma_stdev = np.empty([len(returns) - period_int, ])
        stndr_data = pd.Series(index=returns.index)

        # for efficiency, dont do it in the loop
        sqrd_returns = returns ** 2

        # computations here happen in different times, bc we first need all the ewma_stdev
        # first get the stdev according to the ewma
        for i in range(len(returns) - period_int):
            if i == 0:
                sqrd_data = sqrd_returns[-period_int:]
            else:
                sqrd_data = sqrd_returns[-period_int - i:-i]
            ewma_stdev[-i - 1] = math.sqrt(sum(weights * sqrd_data))

        # get the standardized data by dividing for the ewma_stdev
        # length is here -1 bc we standardize by the ewma_stdev of the previous period
        # hence also ewma_stdev is [-i - 2] instead of [-i - 1]
        for i in range(len(returns) - period_int - 1):
            stndr_data[-i - 1] = returns[-i - 1] / ewma_stdev[-i - 2]
        # NON si ottiene lo stesso numero di osservazioni come negli altri casi, perché ewma_stdev è NA per le prime
        # (period_int - 1) osservazioni
        stndr_data = stndr_data[pd.notnull(stndr_data)]
        # get the percentile and unfilter back the data
        for i in range(len(stndr_data) - period_int):
            if i == 0:
                stndr_data2 = stndr_data[-period_int:]
            else:
                stndr_data2 = stndr_data[-period_int - i:-i]

                stndr_data_pct = np.percentile(stndr_data2, 1 - conf_int)

                # unfilter back with the current stdev
                VaR[-i - 1] = -(stndr_data_pct * ewma_stdev[-i - 1])

        # for FHS the single take of VaR does not work bc we need to standardize for the preceeding stdev
        # hence it is always necessary to calculate the whole series and take the last value
        if series:
            VaR = VaR
        else:
            VaR = VaR[-1]

    return VaR


def compare_VaR(returns, conf_int=.95, period_int=100, ewma_discount_f=.94):
    """
    https://github.com/BSIC/VaR/blob/master/VaR.py
    """
    # call the single VaR series
    VaRPN = compute_VaR(returns, formula="Parametric Normal", conf_int=conf_int, period_int=period_int,
                        ewma_discount_f=ewma_discount_f, series=True, removeNA=True)
    VaREWMA = compute_VaR(returns, formula="Parametric EWMA", conf_int=conf_int, period_int=period_int,
                          ewma_discount_f=ewma_discount_f, series=True, removeNA=True)
    VaRHS = compute_VaR(returns, formula="Historical Simulation", conf_int=conf_int, period_int=period_int,
                        ewma_discount_f=ewma_discount_f, series=True, removeNA=True)
    VaRFHS = compute_VaR(returns, formula="Filtered Historical Simulation", conf_int=conf_int, period_int=period_int,
                         ewma_discount_f=ewma_discount_f, series=True, removeNA=True)

    # concat the different VaR series in the same dataframe and plot it
    AllVaR = pd.concat([VaRPN, VaREWMA, VaRHS, VaRFHS], axis=1)
    AllVaR.plot(lw=1)

    return AllVaR


def ewstats(returns, decay=None, window=None):
    """
    Replica funzione ewstats di Matlab
    https://it.mathworks.com/help/finance/ewstats.html?s_tid=doc_ta
    :param returns:
    :param decay:
    :param window:
    :return:
    """

    # EWSTATS Expected return and covariance from return time series.
    #   Optional exponential weighting emphasizes more recent data.
    #
    #   [ExpReturn, ExpCovariance, NumEffObs] = ewstats(RetSeries, ...
    #                                           DecayFactor, WindowLength)
    #
    #   Inputs:
    #     RetSeries : NUMOBS by NASSETS matrix of equally spaced incremental
    #     return observations.  The first row is the oldest observation, and the
    #     last row is the most recent.
    #
    #     DecayFactor : Controls how much less each observation is weighted than its
    #     successor.  The k'th observation back in time has weight DecayFactor^k.
    #     DecayFactor must lie in the range: 0 < DecayFactor <= 1.
    #     The default is DecayFactor = 1, which is the equally weighted linear
    #     moving average Model (BIS).
    #
    #     WindowLength: The number of recent observations used in
    #     the computation.  The default is all NUMOBS observations.
    #
    #   Outputs:
    #     ExpReturn : 1 by NASSETS estimated expected returns.
    #
    #     ExpCovariance : NASSETS by NASSETS estimated covariance matrix.
    #
    #     NumEffObs: The number of effective observations is given by the formula:
    #     NumEffObs = (1-DecayFactor^WindowLength)/(1-DecayFactor).  Smaller
    #     DecayFactors or WindowLengths emphasize recent data more strongly, but
    #     use less of the available data set.
    #
    #   The standard deviations of the asset return processes are given by:
    #   STDVec = sqrt(diag(ECov)).  The correlation matrix is :
    #   CorrMat = VarMat./( STDVec*STDVec' )
    #
    #   See also MEAN, COV, COV2CORR.

    n_obs = len(returns)
    n_assets = len(returns.columns)

    # size the series and the window
    if window is None:
        window = n_obs

    if decay is None:
        decay = 1

    if decay <= 0 or decay > 1:
        raise ValueError("Must have 0 < decay factor <= 1")

    if window > n_obs:
        raise ValueError(f"Window Length {window} must be <= number of observations {n_obs}")

    # ------------------------------------------------------------------------
    # size the data to the window
    returns = returns.iloc[n_obs - window:n_obs, :]

    # Calculate decay coefficients
    decay_powers = np.arange(window, 0, -1)
    var_wgt = np.sqrt(decay) ** decay_powers
    ret_wgt = (decay) ** decay_powers

    n_eff = sum(ret_wgt)  # number of equivalent values in computation

    pd_ret_wgt = pd.Series(ret_wgt, index=returns.index)
    pd_var_wgt = pd.Series(var_wgt, index=returns.index)

    # Compute the exponentially weighted mean return
    ret_ewm = returns.multiply(pd_ret_wgt, axis=0)
    # WtSeries <- matrix(rep(RetWts, times = NumSeries),
    #                    nrow = length(RetWts), ncol = NumSeries) * RetSeries

    Eret = ret_ewm.sum(axis=0) / n_eff

    # Subtract the weighted mean from the original Series
    centered_ret = returns.copy()
    for col in returns.columns:
        centered_ret[col] = returns[col] - Eret[col]
    # CenteredSeries <- RetSeries - matrix(rep(ERet, each = WindowLength),
    #                                      nrow = WindowLength, ncol = length(ERet))

    # Compute the weighted variance
    var_ewm = centered_ret.multiply(pd_var_wgt, axis=0)

    Ecov = var_ewm.T.dot(var_ewm) / n_eff

    return Eret, Ecov, n_eff


def cov2corr(covar):
    """
    Replica la funzione cov2corr di Matlab
    https://it.mathworks.com/help/finance/cov2corr.html
    :param covar:
    :return:
    """
    # version with for-cycles
    # n_obs = len(covar)
    # n_assets = len(covar.columns)
    #
    # sigma = pd.Series(index=covar.index, name="Expected Sigma")
    # corr = pd.DataFrame(columns=covar.columns, index=covar.index)
    #
    # for i in range(n_obs):
    #     sigma.iloc[i] = np.sqrt(covar.iloc[i, i])
    # for i in range(n_obs):
    #     for j in range(n_assets):
    #         corr.iloc[i, j] = covar.iloc[i, j] / (sigma.iloc[i] * sigma.iloc[j])

    # version with matrix computation
    # https://github.com/CamDavidsonPilon/Python-Numerics/blob/master/utils/cov2corr.py
    sigma = np.sqrt(np.diag(covar))
    corr = (covar.T / sigma).T / sigma
    sigma = pd.Series(sigma, index=covar.index, name="Expected Sigma")
    corr = pd.DataFrame(corr, index=covar.index, columns=covar.columns)

    return sigma, corr


# PRINT NICE STATISTICS
def print_stats(ts, line_length=50):
    """
    given pd.Series or pd.DataFrame of time-series (prices), prints statistics
    :param ts: ts.index must contain dates
    :param line_length: int, lenght of line print
    :return:
    """
    if isinstance(ts, pd.Series):
        is_series = True
    elif isinstance(ts, pd.DataFrame):
        is_series = False
    else:
        raise TypeError(f"ts must be pd.Series, pd.DataFrame.")

    # compute returns
    ret = compute_returns(ts)

    def _nice(line, length=50):
        # print nice line given lenght.
        assert isinstance(line, str)
        if line[:2] != "# ":
            line = "# " + line
        print(f"{line}{' ' * max(0, (length - len(line) - 2))} #")

    print("#" * line_length)
    if is_series:
        _nice(f"Timeseries: {ts.name}", line_length)
    else:
        _nice(f"Timeseries: {', '.join(ts.columns)}", line_length)
    _nice(f"# Period: {ts.index.min().strftime('%Y-%m-%d')} - {ts.index.max().strftime('%Y-%m-%d')}", line_length)
    print("#" * line_length)
    if is_series:
        _nice(f"CAGR: {compute_cagr(ts):.2%}", line_length)
        _nice(f"All period return: {compute_allperiod_returns(ts):.2%}", line_length)
        _nice(f"Volatility: {compute_annualized_volatility(ret):.2%}", line_length)
        _nice(f"Sharpe Ratio: {compute_sharpe_ratio(ret):.2}", line_length)
        _nice(f"Sortino Ratio: {compute_sortino_ratio(ts):.2}", line_length)
        _nice(f"Calmar Ratio: {compute_calmar_ratio(ts):.2}", line_length)
        _nice(f"Max Drawdown: {compute_max_drawdown(ts):.2%}", line_length)
        _nice(f"Parametric 95% VaR: {compute_parametric_var(ret):.2%}", line_length)
        _nice(f"Historical 95% VaR: {compute_historical_var(ret):.2%}", line_length)

        print("#" * line_length)
    else:
        _nice(f"That's all for now folks")
        print("#" * line_length)


