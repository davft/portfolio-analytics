# https://github.com/chrisconlan/algorithmic-trading-with-python/blob/master/src/pypm/metrics.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.optimize import root_scalar
import warnings
import datetime
# https://stackoverflow.com/questions/16004076/python-importing-a-module-that-imports-a-module
from . import dates as dt


def rebase_ts(prices, level=100):
    """
    rebases prices time series to V0
    :param prices: pd.Series or pd.DataFrame
    :param level: rebase to level
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("`prices` must be either pd.Series or pd.DataFrame")

    if isinstance(prices, pd.Series):
        if np.isnan(prices.iloc[0]):
            # se la prima osservazione è NaN
            ts = prices.dropna()
        else:
            ts = prices

        ts_rebased = ts / ts.iloc[0] * level

    else:
        # case 2: isinstance(prices, pd.DataFrame)
        if any(prices.iloc[0].isna()):
            # se vi è almeno un NaN nella prima osservazione delle serie storiche
            ts_rebased = list()
            for col in prices.columns:
                ts = prices[col].dropna()
                ts_rebased.append(ts / ts.iloc[0] * level)
            ts_rebased = pd.concat(ts_rebased, axis=1)
        else:
            ts_rebased = prices / prices.iloc[0] * level

    # nel caso in cui ci siano NaN, la serie storica in output potrebbe avere un indice diverso rispetto a prices
    ts_rebased = ts_rebased.reindex(prices.index)

    return ts_rebased


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
            raise ValueError("`freq` can be: 'weekly', 'monthly', 'yearly', or 'w', 'm', 'y'")
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
        samples = np.random.choice(range(len(ret)), size=samples[0:2])
        return bootstrap_ts(ret, samples=samples, index_dates=index_dates)
    else:
        raise Exception(f"samples must be int or np.array")

    # add zeros as first obs to compute prices time series easly later
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


def compute_tracking_error_vol(ts, bmk_ts):
    """
    compute tracking error volatility wrt a benchmark timeseries
    input MUST be returns time-series
    :param ts: pd.Series or pd.DataFrame (returns)
    :param bmk_ts: pd.Series with benchmark returns
    """
    if not isinstance(ts, (pd.Series, pd.DataFrame)):
        print("`ts` must be pd.Series or pd.DataFrame")
        return
    if not isinstance(bmk_ts, pd.Series):
        print("`bmk_ts` must be pd.Series")
        return

    excess_return = compute_excess_return(ts=ts, bmk_ts=bmk_ts)
    tev = compute_annualized_volatility(excess_return)

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

    return ptf_cagr - bmk_cagr


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


def compute_sharpe_ratio(prices, benchmark_rate=0):
    """
    Calculates the sharpe ratio given a price series. Defaults to benchmark_rate
    of zero.
    """
    cagr = compute_cagr(prices)
    returns = compute_returns(prices)
    volatility = compute_annualized_volatility(returns)
    return (cagr - benchmark_rate) / volatility


def compute_information_ratio(prices, bmk_prices):
    """
    Compute the Information ratio wrt a benchmark
    IR = (ptf_returns - bmk_returns) / tracking_error_vol
    :param prices:
    :param bmk_prices:
    :return:
    """
    # compute active return
    active_ret = compute_active_return(prices, bmk_prices)
    # compute TEV
    ptf_ret = compute_returns(prices)
    bmk_ret = compute_returns(bmk_prices)
    tev = compute_tracking_error_vol(ptf_ret, bmk_ret)

    return active_ret / tev


def compute_rolling_sharpe_ratio(prices, n=20, method="simple"):
    """
    Compute an *approximation* of the sharpe ratio on a rolling basis.
    Intended for use as a preference value.
    """
    rolling_return_series = compute_returns(prices, method=method).rolling(n)
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
        bmk_ret = bmk_ret.to_frame(bmk_ret.name)

    # bisogna fare in modo che le date (.index) coincidano
    if len(set(ret.index).symmetric_difference(bmk_ret.index)):
        # merge per allineare le date
        tmp = pd.concat([ret, bmk_ret], axis=1)
        # inserisci zero al posto degli NA
        tmp = tmp.fillna(0)
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
        # serve lavorare su un unico pd.DataFrame
        tmp = pd.concat([ret, bmk_ret], axis=1)
        # inserisci zero al posto degli NA
        tmp = tmp.fillna(0)
        for jcol in bmk_ret.columns:
            m = np.cov(tmp[[*ret.columns, jcol]], rowvar=False)
            variance = m[(m.shape[0] - 1), (m.shape[1] - 1)]
            # non considerare la covarianza con il benchmark
            covariance = m[:-1, (m.shape[1] - 1)]
            # salva beta
            beta[jcol] = covariance / variance

    if sum(beta.shape) == 2:
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


# def compute_beta(ret, bmk_ret, use_linear_reg=False, single_reg=True):
#     """
#     https://corporatefinanceinstitute.com/resources/knowledge/finance/beta-coefficient/
#     :param ret: stock/portfolio returns. works with multiple timeseries as well
#     :param bmk_ret: benchmark returns
#     :param use_linear_reg: boolean -> compute beta using linear regression?
#     :param single_reg: boolean. if True, linear reg of one var, otherwise multivar linear reg
#     :return:
#     """
#     
#     if not isinstance(ret, (pd.Series, pd.DataFrame)):
#         raise TypeError("`ret` must be pd.Series or pd.DataFrame containing returns")
#     if not isinstance(bmk_ret, pd.Series):
#         raise TypeError("`bmk_ret` must be pd.Series containing benchmark returns")
#     
#     # metti insieme le due serie storiche per considerare le stesse date
#     ts_ret = pd.concat([ret, bmk_ret], axis=1)
#     # rimuovi NA
#     ts_ret = ts_ret.fillna(0)
#     if (len(ts_ret) != len(ret)) or (len(ts_ret) != len(bmk_ret)):
#         print("Dropping some obs (`ret` and `bmk_ret` do not have the same index)."
#               " Please check manually if this is okay for you.")
# 
#     if use_linear_reg:
#         # use linear regression to compute beta
#         # controlla se il codice fa effettivamente quanto richiesto
#         m = LinearRegression(fit_intercept=True)
#         if isinstance(ret, pd.Series):
#             m.fit(ts_ret.iloc[:, [0]], ts_ret.iloc[:, [-1]])
#             beta = m.coef_[0][0]
#         else:
#             if single_reg:
#                 # fai la regressione singola di ogni stock/ptf vs bmk
#                 beta = list()
#                 for icol in range(len(ret.columns)):
#                     m.fit(ts_ret.iloc[:, [icol]], ts_ret.iloc[:, [-1]])
#                     beta.append(m.coef_[0][0])
#             else:
#                 # usa regressione multipla
#                 m.fit(ts_ret[:, :-1], ts_ret.iloc[:, [-1]])
#                 beta = m.coef_[0]
#     else:
#         # use variance - covariance
#         # variance and covariance
#         m = np.cov(ts_ret, rowvar=False)
#         variance = m[(m.shape[0] - 1), (m.shape[1] - 1)]
#         # non considerare la covarianza con il benchmark
#         covariance = m[:-1, (m.shape[1] - 1)]
#         beta = covariance / variance
#     
#     if isinstance(ret, pd.DataFrame):
#         beta = pd.Series(beta, index=ret.columns)
#         
#     return beta
    

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


def compute_sortino_ratio(prices, benchmark_rate=0):
    """
    Calculates the sortino ratio.
    """
    cagr = compute_cagr(prices)
    return_series = compute_returns(prices)
    downside_deviation = compute_annualized_downside_deviation(return_series)
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


def compute_calmar_ratio(prices, years_past=3, method="simple"):
    """
    Return the percent max drawdown ratio over the past n years, otherwise
    known as the Calmar Ratio (3yr)
    """

    # Filter series on past three years
    last_date = prices.index[-1]
    n_years_ago = last_date - pd.Timedelta(days=years_past * 365.25)
    series = prices[prices.index > n_years_ago]

    # Compute annualized percent max drawdown ratio
    percent_drawdown = -compute_max_drawdown(series, method=method)
    cagr = compute_cagr(series)
    return cagr / percent_drawdown


def compute_parametric_var(returns, conf_lvl=.95):
    """
    :param returns
    :param conf_lvl: float, confidence level
    ref: https://blog.quantinsti.com/calculating-value-at-risk-in-excel-python/
    """
    mu = np.mean(returns)
    sigma = np.std(returns)

    VaR = norm.ppf(1 - conf_lvl, mu, sigma)
    return VaR


def compute_historical_var(returns, conf_lvl=.95):
    """
    :param returns
    :param conf_lvl: float, confidence level
    ref: https://blog.quantinsti.com/calculating-value-at-risk-in-excel-python/
    """
    VaR = returns.quantile(1 - conf_lvl)
    return VaR


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
        print("please provide argument `df` (can't be None if `ts` is pd.Series)")
        return

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
            for i in range(len(ts.columns) - 1):
                pair_corr = ts[ts.columns[i]].rolling(window).corr(df).to_frame()
                pair_corr.columns = [ts.columns[i] + "-" + df.name]
                corr.append(pair_corr)

            corr = pd.concat(corr, axis=1)

    corr = corr.dropna()
    return corr


def compute_corr(ts, df=None):
    """
    computes correlation over all given period
    if `ts` is pd.Series, then it computes `ts` vs all `df` cols
    if `ts` is pd.DataFrame and `df` is None, then it computes all pairwise corr between `ts` cols
    if `ts` is pd.DataFrame and `df` is pd.Series, then it computes all `ts` vs `df`
    :param ts: pd.Series or pd.DataFrame
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
            corr = pd.DataFrame(0, index=ts.columns, columns=ts.columns)
            for i in range(len(ts.columns)):
                for j in range(i, len(ts.columns)):
                    corr.iloc[i, j] = ts[ts.columns[i]].corr(ts[ts.columns[j]])
                    # it's the same
                    corr.iloc[j, i] = corr.iloc[i, j]

        elif isinstance(df, pd.Series):
            corr = ts.apply(lambda x: x.corr(df), axis=0).reset_index(name="corr")

        elif isinstance(df, pd.DataFrame):
            corr = list()
            for i in range(len(df.columns)):
                pair_corr = ts.apply(lambda x: x.corr(df[df.columns[i]]), axis=0)  # .reset_index(name=df.columns[i])
                pair_corr.name = df.columns[i]
                corr.append(pair_corr)
            corr = pd.concat(corr, axis=1)

    return corr


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
        turnover_i.append(np.sum(np.abs(hh.iloc[i] - hh.iloc[i - 1])) / 2)
    
    turnover = pd.Series(turnover_i, index=hh.index[1:])

    return turnover


# REPLICA
def compute_replica_groupby(holdings, groupby=["country", "sector"], overall_coverage=.85, groupby_coverage=None,
                            threshold=.02, rebase=1, minw=0, maxw=1, id_col="isin", w_col="w", keep_all_cols=True):
    """
    Compute replica of a portfolio (could be index), grouping on key specified by `groupby`
    :param holdings: pd.DataFrame with components for a single date
    :param groupby: key for the index replication. must be cols of holdings. str or list of str
    :param overall_coverage: percentage of index covering. float
    :param groupby_coverage: None or float. if float, then also check for coverage inside groupby.
        Makes sense to be used if index is weighted by marketcap
    :param threshold: minimum weight a groupby cell must have in order to get more than 1 stock. float
    :param rebase: int/float or None. if None, do not rebase. if not None, rebase to value of rebase
    :param minw: float, minimum weight
    :param maxw: float, maximum weight
    :param id_col: str, column of holdings. col of the identifier
    :param w_col: str, column of holdings. col of the weights
    :param keep_all_cols: boolean, if True keep all columns from initial holdings
    :return replica portfolio
    """

    # check if inputs are of the correct type
    if not isinstance(holdings, pd.DataFrame):
        raise TypeError("holdings must be pd.DataFrame")
    if not isinstance(groupby, (str, list)):
        raise TypeError("groupby must be str or list of str")
    if isinstance(groupby, list):
        if not all(isinstance(item, str) for item in groupby):
            raise TypeError("all elements of groupby must be str")
    else:
        groupby = [groupby]
    if not isinstance(overall_coverage, (int, float)):
        raise TypeError("overall_coverage must be a float")
    if groupby_coverage is not None:
        if not isinstance(groupby_coverage, float):
            raise TypeError("groupby_coverage must be None or float")
    if not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a float")
    if rebase is not None:
        if not isinstance(rebase, (int, float)):
            raise TypeError("groupby_coverage must be None or int/float")
    assert isinstance(minw, (int, float)), "`minw` must be float"
    assert isinstance(maxw, (int, float)), "`maxw` must be float"
    if (minw < 0) | (minw >= 1):
        raise ValueError("`minw` in [0, 1) required")
    if (maxw <= 0) | (maxw > 1):
        raise ValueError("`maxw` in (0, 1] required")
    if not isinstance(id_col, str):
        raise TypeError("id_col must be str")
    if not isinstance(w_col, str):
        raise TypeError("w_col must be str")
    if not isinstance(keep_all_cols, bool):
        raise TypeError("keep_all_cols must be boolean")

    # check if all needed columns are in holdings
    required_cols = [id_col, w_col, *groupby]
    if not set(required_cols).issubset(set(holdings.columns)):
        raise Exception(f"components must have columns: {id_col}, {w_col}, {groupby}")

    def cut_coverage(df, col, lvl):
        """
        :param df: pd.DataFrame with col in df.columns
        :param col: str, df's column of floats
        :param lvl: float, threshold where to cut
        """
        x = df[col].values
        # https://stackoverflow.com/questions/2361426/get-the-first-item-from-an-iterable-that-matches-a-condition
        cut_idx = next((i for i, y in enumerate(x) if y > lvl), 1)
        return df.iloc[0:cut_idx]

    if rebase is not None:
        # rebase weights
        holdings[w_col] = holdings[w_col] / (holdings[w_col].sum() * rebase)

    if keep_all_cols:
        # mantieni da parte le colonne non necessarie, fai il merge in fondo
        other_cols = list(set(holdings.columns).difference(required_cols))
        other_cols.append(id_col)
        if len(other_cols) > 1:
            df_other_cols = holdings[other_cols]

    # select only needed columns
    hldg = holdings[required_cols]
    stat = hldg.groupby(groupby).sum()
    # sort descending by weight
    stat = stat.sort_values(by=[w_col], ascending=False)
    # compute cumulative sum
    stat["cumulative_sum"] = stat[w_col].cumsum()
    stat = stat.reset_index()
    # select groupby cells up to `overall_coverage`
    cells = cut_coverage(stat, col="cumulative_sum", lvl=overall_coverage)
    cells = cells.copy()
    print(f"Selecting first {len(cells)} out of {len(stat)} combinations of {groupby}")

    # rebase weight to 1 on selected cells
    cells["w_rebase"] = cells[w_col] / cells[w_col].sum()

    # assegna il numero di titoli da scegliere all'interno di ogni combinazione groupby
    cells["n_stocks"] = np.ceil(cells["w_rebase"] / threshold)

    if groupby_coverage is None:
        # first method
        # gli n assegnati NON vengono modificati
        # vengono scelte le prime n stock per weight in ogni combinazione groupby

        # attacca a `hldg` le informazioni sulla numerosità delle coppie sector/country
        hldg = pd.merge(hldg, cells[[*groupby, "n_stocks"]], how="left", on=groupby)

        # sort by groupby and weight, ascending order for groupby and descending for weight
        hldg = hldg.sort_values([*groupby, w_col], ascending=[*[True] * len(groupby), False])

        # NA sono le celle non scelte
        replica = hldg.dropna()
        # select first n_stock for each cells
        # select first x.shape[0] if n_stock > x.shape[0]
        replica = replica.groupby(groupby).apply(lambda x: x.iloc[0:min(int(x["n_stocks"].unique()), x.shape[0])])
        replica = replica.reset_index(drop=True)

    else:
        # second method
        # es: per sector == 'a' & country == 'A' ho n = 2, e le seguenti stock:
        # | ticker | w   |
        # | aa     | .15 |
        # | bb     | .12 |
        # | cc     | .10 |
        # | dd     | .09 |
        # | ee     | .08 |
        # con METODO 1 verrebbero scelte le stock `aa` e `bb`, poiché le prime 2 (n = 2)
        # per marketcap
        # con METOOD 2 vengono scelte le stock `aa`, `bb` e `cc` per rispettare la
        # `copertura_groupby` = .7
        # tot_w = .15 + .12 + .1 + .09 + .08 = .54
        # tot_w * copertura_groupby = .378 <-- quando la cumulata dei pesi raggiunge .378 TAGLIA !
        # cumsum(w) = .15, .27, .37, .46, .54
        # tagliamo al terzo elemento, per cui la `cumsum` raggiunge il massimo valore
        # che è minore di .378

        # coppie groupby da cui devo pescare k stocks s.t. venga replicato almeno
        # copertura_groupby% della capitalizzazione della coppia
        cells_k = cells.loc[cells["n_stocks"] > 1]
        hldg_k = pd.merge(hldg, cells_k[groupby], how="inner", on=groupby)
        # sort by groupby and weight, ascending order for groupby and descending for weight
        hldg_k = hldg_k.sort_values([*groupby, w_col], ascending=[*[True] * len(groupby), False])

        # calcola il peso cumulato delle combinazioni groupby
        hldg_k["cumulative_sum"] = \
            hldg_k.groupby(groupby).apply(lambda x: x[w_col].cumsum() / x[w_col].sum()).values
        # prendi le prime k stocks s.t. vi è una copertura almeno del copertura_groupby%
        replica = hldg_k.groupby(groupby).apply(lambda x: cut_coverage(x, col="cumulative_sum", lvl=groupby_coverage))
        replica = replica.reset_index(drop=True)
        del replica["cumulative_sum"]

        # combinazioni groupby da cui devo pescare solo 1 stock
        cells_k = cells.loc[cells["n_stocks"] == 1]
        if len(cells_k):
            hldg_k = pd.merge(hldg, cells_k[groupby], how="inner", on=groupby)
            # sort by groupby and weight, ascending order for groupby and descending for weight
            hldg_k = hldg_k.sort_values([*groupby, w_col], ascending=[*[True] * len(groupby), False])

            # prendi le prime stocks in ogni combinazione ed aggiungile a ptf
            replica = replica.append(
                hldg_k.groupby(groupby).apply(lambda x: x.iloc[0]).reset_index(drop=True)
            )

    print(f"Selected {replica.shape[0]} stocks out of {holdings.shape[0]}")

    # set replica weights
    # aggiungi colonna contente il peso di ogni combinazione groupby
    replica = pd.merge(replica, cells[[*groupby, "w_rebase"]], how="left", on=groupby)
    replica["w_replica"] = \
        replica.groupby(groupby).apply(lambda x: x[w_col] / x[w_col].sum() * x["w_rebase"]).values

    output_cols = [*required_cols, "w_replica"]
    replica = replica[output_cols]

    if keep_all_cols:
        replica = replica.merge(df_other_cols, how="left", on=[id_col])

    # drop w_col (corresponds to original index weight), rename "w_replica" to w_col (easier to use later in the script)
    del replica[w_col]
    replica = replica.rename(columns={"w_replica": w_col})

    # check if weights respect bounds, if not rebase them
    # perform this operation on a groupby-level, in order to maintain the same overall exposure
    replica = replica.groupby(groupby)\
        .apply(lambda x: redistribuite_weights_given_bounds(df=x, minw=minw, maxw=maxw, w_col=w_col))\
        .reset_index(drop=True)

    return replica


def compute_lux_replica(hldg, N=100, groupby="GICS Sector", agg=None, w_col="weight", id_col="ISIN",
                        equally_weighted=True, rebase=1.):
    """
    Replica di un index/ETF/ptf utilizzando la metodologia di Lux:
        - aggregazione per le colonne indicate in groupby
        - calcola il numero di titoli per ogni cella, problema di root finding per arrivare ad N titoli
        - calcola il peso dato aggregazione cella / numero titoli per ogni cella
        - seleziona il numero di stock richiesto ed assegna il peso di cui sopra
        - si ottiene quindi una replica equipesata all'interno di ogni cella
    :param hldg: pd.DataFrame contenente le holdings riferite ad una sola data
    :param N: int, numero finale di titoli che si vogliono ottenere nella replica
    :param groupby: str or list, colonne da usare per l'aggregazione
    :param agg: None or pd.DataFrame, serve per ottenere una scomposizione su groupby CUSTOM
        - es/ per replica di TRVCI si parte da composizioni SPX, si aggrega per settore, e si aggiustano i pesi
            settoriali utilizzando il beta settoriale vs TRVCI. si calcola fuori dalla funzione e lo si da in input
    :param w_col: str, nome della colonna contenente i pesi di ogni titolo nel ptf iniziale
    :param id_col: str, nome della colonna contenete ISIN, ticker, CUSIP (identifier degli strumenti)
    :param equally_weighted: boolean. if False, distribuisci i pesi in base al peso iniziale in hldg
    :param rebase: int/float or None. if None, do not rebase. if not None, rebase to value of rebase
    :return: replica, pd.DataFrame
    """

    if not isinstance(groupby, (str, list)):
        raise TypeError("`groupby` must be str or list")
    if isinstance(groupby, str):
        groupby = [groupby]

    if not isinstance(w_col, str):
        raise TypeError("`w_col` must be str")
    if not isinstance(id_col, str):
        raise TypeError("`id_col` must be str")

    if rebase is not None:
        if not isinstance(rebase, (int, float)):
            raise TypeError("groupby_coverage must be None or int/float")

    # columns required for computing replica portfolio
    required_cols = [id_col, w_col, *groupby]
    if not set(required_cols).issubset(hldg.columns):
        raise Exception(f"`hldg` must have columns: {id_col}, {w_col}, {groupby}")

    if rebase is not None:
        # rebase weights
        hldg[w_col] = hldg[w_col] / (hldg[w_col].sum() * rebase)

    if agg is None:
        # calcolare agg se non è fornito
        agg = hldg.groupby(groupby)[w_col].sum().reset_index(name="w_groupby")
    else:
        if not {*groupby, "w_groupby"}.issubset(agg.columns):
            raise Exception(f"`agg` must have columns: {groupby}, w_groupby")

    def find_N_fittizio(x, agg, N):
        N_tmp = [round(x * y, 0) for y in agg["w_groupby"]]
        return sum(N_tmp) - N

    N_fittizio = root_scalar(find_N_fittizio, args=(agg, N), bracket=[0, N + 50], x0=N).root

    # calcola il numero di stock che vanno selezionate per ogni combinazione groupby
    agg["n_stocks"] = [int(round(N_fittizio * x, 0)) for x in agg["w_groupby"]]
    
    # rimuovi le combinazioni con 0 stock
    agg = agg[agg["n_stocks"] != 0]


    # se c'è GOOGL US (US02079K3059) rimuovi GOOG US (US02079K1079)
    if "US02079K3059" in hldg[id_col]:
        hldg = hldg[hldg[id_col] != "US02079K1079"]

    # sort by groupby and weight, ascending order for groupby and descending for weight
    hldg = hldg.sort_values([*groupby, w_col], ascending=[*[True] * len(groupby), False])

    # aggiungi info su n_stock a hldg
    hldg = hldg.merge(agg[[*groupby, "n_stocks"]], how="left", on=groupby)
    # NA sono le celle non scelte
    replica = hldg.dropna()
    # seleziona i titoli che vanno scelti
    replica = replica.groupby(groupby).apply(lambda x: x.iloc[0:min(int(x["n_stocks"].unique()), x.shape[0])])
    replica = replica.reset_index(drop=True)

    # set replica weights
    if equally_weighted:
        # calcola il peso (equally weighted) di ogni stock all'interno del groupby
        agg["w_stock"] = agg["w_groupby"] / agg["n_stocks"]
        # aggiungi colonna contente il peso di ogni combinazione groupby
        replica = replica.merge(agg[[*groupby, "w_stock"]], how="left", on=groupby)
        replica["w_replica"] = replica["w_stock"]
    else:
        # aggiungi il peso della combinazione groupby
        replica = replica.merge(agg[[*groupby, "w_groupby"]], how="left", on=groupby)
        replica["w_replica"] = \
            replica.groupby(groupby).apply(lambda x: x[w_col] / x[w_col].sum() * x["w_groupby"]).values


    output_cols = [id_col, w_col, *groupby, "w_replica"]
    replica = replica[output_cols]

    return replica


def compute_groupby_difference(hldg, bmk_hldg, groupby="GICS Sector", w_col="Weight"):
    """
    compute difference between aggregation of ptf vs index
    :param hldg: pd.DataFrame with portfolio composition [only one ptf per call]
    :param bmk_hldg: pd.DataFrame with benchmark composition
    :param groupby: list of keys to groupby
    :param w_col: str, weight colname
    # :param verbose: boolean, if True return tuples (diff, ptf_exp, bmk_exp)
    """
    if not isinstance(hldg, pd.DataFrame):
        raise TypeError("`hldg` must be a pd.DataFrame containing portfolio compositions")
    if not isinstance(bmk_hldg, pd.DataFrame):
        raise TypeError("`bmk_hldg` must be a pd.DataFrame containing portfolio compositions")
    if not isinstance(groupby, (str, list)):
        raise TypeError("`groupby` must be str or list of str with column names")
    if isinstance(groupby, str):
        groupby = [groupby]
    if not isinstance(w_col, str):
        raise TypeError("w_col must be str")

    # # columns to lowercase
    # hldg.columns = hldg.columns.map(lambda x: x.lower())
    # bmk_hldg.columns = bmk_hldg.columns.map(lambda x: x.lower())
    # groupby = list(map(lambda x: x.lower(), groupby))


    # group by `groupby` and compute allocation weights
    fun_exposure = lambda x: x[w_col].sum()
    ptf_exposure = hldg.groupby(groupby).apply(fun_exposure).reset_index(name="replica_exp")
    bmk_exposure = bmk_hldg.groupby(groupby).apply(fun_exposure).reset_index(name="bmk_exp")

    diff = pd.merge(ptf_exposure, bmk_exposure, how="outer", on=groupby)
    diff = diff.fillna(0)
    diff["difference"] = diff["replica_exp"] - diff["bmk_exp"]

    return diff


def get_ew_weights(ptf, groupby=("data_val", "index_ticker"), on="ticker", index="data_val"):
    """
    Get pd.DataFrame to be used as `weights` for `Portfolio()`
    the function only works with ALL input. Must be extended to allows groupby=None and index=None
    :param ptf: pd.DataFrame with cols at least (groupby, on, index)
    :param groupby: at what level applies the equally weighted portfolio
    :param on: column of `ptf`, ideally "ticker" or "ISIN"
    :param index: columns of `ptf` with dates
    :return:
    """
    # add column with equally weighted weights at the `groupby` level
    ptf["weights"] = ptf.groupby(groupby)[on].transform(lambda x: 1 / len(x))
    # pivot to get the desired shape
    ew_weights = ptf.pivot(index=index, columns=on, values="weights")
    # fill NAs with 0
    ew_weights = ew_weights.fillna(0)
    return ew_weights


def redistribuite_weights_given_bounds(df, minw=0, maxw=1, w_col="Weight"):
    """
    Check if weights in df respect bounds [minw, maxw]. if yes return df
    If not, squeeze distribution (if possibile) to make weights respect bounds
    :param df: pd.DataFrame, containing at least w_col column
    :param minw: float [0, 1)
    :param maxw: float (0, 1]
    :param w_col: str, name of column containing weights
    :return:
    """

    assert isinstance(df, pd.DataFrame), "`df` must be pd.DataFrame"
    assert isinstance(minw, (int, float)), "`minw` must be float"
    assert isinstance(maxw, (int, float)), "`maxw` must be float"
    assert isinstance(w_col, str), "`w_col` must be str"
    if (minw < 0) | (minw >= 1):
        raise ValueError("`minw` in [0, 1) required")
    if (maxw <= 0) | (maxw > 1):
        raise ValueError("`maxw` in (0, 1] required")

    # check if bounds are respected
    if (min(df[w_col]) >= minw) & (max(df[w_col]) <= maxw):
        # if yes, no adjustment needed
        return df

    # check if operation is feasible
    if sum(df[w_col]) < len(df) * minw:
        raise ValueError(f"infeasible: sum(df[w_col]) must be >= len(df) * minw ({len(df) * minw:.2%}) "
                         f"it is {sum(df[w_col]):.2%}")
    if sum(df[w_col]) > len(df) * maxw:
        raise ValueError(f"infeasible: sum(df[w_col]) must be <= len(df) * maxw: ({len(df) * maxw:.2%}) "
                         f"it is {sum(df[w_col]):.2%}")

    # isolate values not in bound (and on its limits): [0, minw], [maxw, 1]
    out_bound = df[(df[w_col] <= minw) | (df[w_col] >= maxw)]
    while len(out_bound[(out_bound[w_col] < minw) | (out_bound[w_col] > maxw)]):
        # compute how much weight you need to redistribuite across obs in bounds
        # 1. weight needed to bring values below minw to minw (negative quantity)
        excess_w = sum(out_bound[out_bound[w_col] < minw][w_col]) - len(out_bound[out_bound[w_col] < minw]) * minw
        # 2. excess weight you get from bringing values above maxw to maxw (positive quantity)
        excess_w += sum(out_bound[out_bound[w_col] > maxw][w_col]) - len(out_bound[out_bound[w_col] > maxw]) * maxw
        # floor weights to minw and cap to maxw
        out_bound[w_col] = out_bound[w_col].clip(minw, maxw)

        btw_bound = df[(df[w_col] > minw) & (df[w_col] < maxw)]
        btw_bound[w_col] = btw_bound[w_col] + btw_bound[w_col] / sum(btw_bound[w_col]) * excess_w

        # override df with the new weights
        df = pd.concat([btw_bound, out_bound], axis=0)
        # isolate again values not in bound to repeat the cycle if needed
        out_bound = df[(df[w_col] <= minw) | (df[w_col] >= maxw)]

    return df
