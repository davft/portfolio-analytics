import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import warnings


def rebase_ts(prices, level=100):
    """
    rebases prices time series to V0
    :param prices: pd.Series or pd.DataFrame
    :param level: rebase to level
    """
    return prices / prices.iloc[0] * level


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
        raise ValueError("method should be either simple or log")

    if isinstance(prices, pd.Series):
        returns = ret_fun(prices)
    elif isinstance(prices, pd.DataFrame):
        returns = prices.apply(ret_fun, axis=0)
    else:
        raise TypeError("price_series should be either pd.Series or pd.DataFrame")

    return returns


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


def compute_rolling_sharpe_ratio(prices, n=20, method="simple"):
    """
    Compute an *approximation* of the sharpe ratio on a rolling basis.
    Intended for use as a preference value.
    """
    rolling_return_series = compute_returns(prices, method=method).rolling(n)
    return rolling_return_series.mean() / rolling_return_series.std()


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
    compute rolling correlations. if ts is pd.Series, then it computes `ts` vs all `df` cols
    if `ts` is pd.DataFrame, then it computes all pairwise corr between `ts` cols
    :param ts: pd.Series or pd.DataFrame
        if pd.Series, then `df` is mandatory and must be pd.Series or pd.DataFrame
        if pd.DataFrame, then `df` is not needed and all pairwise correlations are computed
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
        corr = list()
        for i in range(len(ts.columns) - 1):
            for j in range(i + 1, len(ts.columns)):
                pair_corr = ts[ts.columns[i]].rolling(window).corr(ts[ts.columns[j]]).to_frame()
                pair_corr.columns = [ts.columns[i] + "-" + ts.columns[j]]
                corr.append(pair_corr)

        corr = pd.concat(corr, axis=1)

    corr = corr.dropna()
    return corr


# REPLICA
def compute_replica_groupby(holdings, groupby=["country", "sector"], overall_coverage=.85, groupby_coverage=None,
                            threshold=.02):
    """
    Compute replica of a portfolio (could be index), grouping on key specified by `groupby`
    :param holdings: pd.DataFrame with components for a single date
    :param groupby: key for the index replication
    :param overall_coverage: percentage of index covering
    :param groupby_coverage: None or float. if float, then also check for coverage inside groupby.
        Makes sense to be used if index is weighted by marketcap
    :param threshold: minimum weight a groupby cell must have in order to get more than 1 stock
    :return replica portfolio
    """

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

    # columns to lowercase
    holdings.columns = holdings.columns.map(lambda x: x.lower())
    groupby = list(map(lambda x: x.lower(), groupby))
    # columns required for computing replica portfolio
    required_cols = ["isin", "w", *groupby]
    if not set(required_cols).issubset(set(holdings.columns)):
        raise Exception("components must have columns: 'isin', 'w', `groupby`")

    # rebase weights to 1
    holdings["w"] = holdings["w"] / holdings["w"].sum()

    # select only needed columns
    hldg = holdings[required_cols]
    stat = hldg.groupby(groupby).sum()
    # sort descending by weight
    stat = stat.sort_values(by=["w"], ascending=False)
    # compute cumulative sum
    stat["cumulative_sum"] = stat["w"].cumsum()
    stat = stat.reset_index()
    # select groupby cells up to `overall_coverage`
    cells = cut_coverage(stat, col="cumulative_sum", lvl=overall_coverage)
    cells = cells.copy()
    print("Selecting first {} out of {} combinations of {}".format(cells.shape[0], stat.shape[0], groupby))

    # rebase weight to 1 on selected cells
    cells["w_rebase"] = cells["w"] / cells["w"].sum()

    # assegna il numero di titoli da scegliere all'interno di ogni combinazione groupby
    cells["n_stocks"] = np.ceil(cells["w_rebase"] / threshold)

    if groupby_coverage is None:
        # first method
        # gli n assegnati NON vengono modificati
        # vengono scelte le prime n stock per weight in ogni combinazione groupby

        # attacca a `hldg` le informazioni sulla numerosità delle coppie sector/country
        hldg = pd.merge(hldg, cells[[*groupby, "n_stocks"]], how="left", on=groupby)

        # sort by groupby and weight, ascending order for groupby and descending for weight
        hldg = hldg.sort_values([*groupby, "w"], ascending=[*[True] * len(groupby), False])

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
        hldg_k = hldg_k.sort_values([*groupby, "w"], ascending=[*[True] * len(groupby), False])

        # calcola il peso cumulato delle combinazioni groupby
        hldg_k["cumulative_sum"] = hldg_k.groupby(groupby).apply(lambda x: x["w"].cumsum() / x["w"].sum()).values
        # prendi le prime k stocks s.t. vi è una copertura almeno del copertura_groupby%
        replica = hldg_k.groupby(groupby).apply(lambda x: cut_coverage(x, col="cumulative_sum", lvl=groupby_coverage))
        replica = replica.reset_index(drop=True)
        del replica["cumulative_sum"]

        # combinazioni groupby da cui devo pescare solo 1 stock
        cells_k = cells.loc[cells["n_stocks"] == 1]
        if len(cells_k):
            hldg_k = pd.merge(hldg, cells_k[groupby], how="inner", on=groupby)
            # sort by groupby and weight, ascending order for groupby and descending for weight
            hldg_k = hldg_k.sort_values([*groupby, "w"], ascending=[*[True] * len(groupby), False])

            # prendi le prime stocks in ogni combinazione ed aggiungile a ptf
            replica = replica.append(
                hldg_k.groupby(groupby).apply(lambda x: x.iloc[0]).reset_index(drop=True)
            )

    print("Selected {} stocks out of {}".format(replica.shape[0], holdings.shape[0]))

    # set replica weights
    # aggiungi colonna contente il peso di ogni combinazione groupby
    replica = pd.merge(replica, cells[[*groupby, "w_rebase"]], how="left", on=groupby)
    replica["w_replica"] = replica.groupby(groupby).apply(lambda x: x["w"] / x["w"].sum() * x["w_rebase"]).values

    output_cols = ["isin", "w", *groupby, "w_replica"]
    replica = replica[output_cols]

    # remove index

    return replica


def compute_groupby_difference(hldg, bmk_hldg, groupby=[], verbose=False):
    """
    compute difference between aggregation of ptf vs index
    :param hldg: pd.DataFrame with portfolio composition [only one ptf per call]
    :param bmk_hldg: pd.DataFrame with benchmark composition
    :param groupby: list of keys to groupby
    :param verbose: boolean, if True return tuples (diff, ptf_exp, bmk_exp)
    """
    if not isinstance(hldg, pd.DataFrame):
        print("`hldg` must be a pd.DataFrame containing portfolio compositions")
        return
    if not isinstance(bmk_hldg, pd.DataFrame):
        print("`bmk_hldg` must be a pd.DataFrame containing portfolio compositions")
        return
    if not len(groupby):
        print("`groupby` must be str or list of str with column names")
        return
    if isinstance(groupby, str):
        groupby = [groupby]

    # group by `groupby` and compute allocation weights
    fun_exposure = lambda x: x["w"].sum()
    ptf_exposure = hldg.groupby(groupby).apply(fun_exposure)
    ptf_exposure.name = "ptf_exp"
    bmk_exposure = bmk_hldg.groupby(groupby).apply(fun_exposure)
    bmk_exposure.name = "bmk_exp"

    diff = pd.merge(ptf_exposure, bmk_exposure, how="outer", left_index=True, right_index=True)
    diff = diff.fillna(0)
    diff["difference"] = diff["ptf_exp"] - diff["bmk_exp"]

    if verbose:
        return diff["difference"], ptf_exposure, bmk_exposure
    else:
        return diff["difference"]
