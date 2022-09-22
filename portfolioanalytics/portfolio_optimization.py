import pandas as pd
import numpy as np
import warnings
from scipy.optimize import minimize, LinearConstraint
# https://stackoverflow.com/questions/16004076/python-importing-a-module-that-imports-a-module
from .portfolio_return import Portfolio as Portfolio
from . import functions as pa


class PortfolioOptimization(Portfolio):
    """

    """
    def __init__(self, prices, benchmark, metric="TEV", rebal_dates=None, lookback_period=None,
                 V0=100, max_leverage=1, method="simple"):
        """
        :param lookback_period: int, number of months to be taken into consideration for optimization
            if None: use all period
        """
        assert isinstance(prices, pd.DataFrame), "`prices` must be pd.DataFrame"
        if rebal_dates is None:
            assert lookback_period is None, "if `rebal_dates` is None, `lookback_period` must be None. Optimization" \
                                            "will be performed on the whole period."
        else:
            assert isinstance(rebal_dates, (list, tuple, pd.Series)), "`rebal_dates` must be list, tuple or pd.Series"
            assert isinstance(lookback_period, int), "if `rebal_dates` is given, `lookback_period` must be int"
        assert metric in ["TEV", "SR", "IR", "Sortino", "Vol", "Return", "RetVsBmk", "Beta", "Beta=1"], \
            "`metric` must be either TEV, SR, IR, Sortino, Vol, RetVsBmk, Beta, Beta=1"

        # Initialize Portfolio class
        # probably you don't need it
        # super().__init__(prices, V0=V0, max_leverage=max_leverage, method=method, benchmark_rate=0)
        self.prices = prices

        # Save information about benchmark
        self.bmk_ts = benchmark
        self.bmk_ret = pa.compute_returns(benchmark)

        # Save rebalancing dates information
        self.lookback = lookback_period
        if rebal_dates is None and lookback_period is None:
            self.rebal_dates = [self.prices.index[-1]]
            self.start_optim_date = self.prices.index[0]
        else:
            if isinstance(rebal_dates, (list, tuple)):
                self.rebal_dates = list(rebal_dates)
            else:
                # case: pd.Series
                self.rebal_dates = list(rebal_dates.values)
            self.start_optim_date = min(self.rebal_dates) - pd.tseries.offsets.DateOffset(months=lookback_period)
        # controlla che le rebalancing dates siano valide
        self.rebal_dates = self.check_rebal_dates()

        # Define the optimization objective (to be minimezed)
        self.metric = metric
        if metric == "TEV":
            self.__class__ = OptimTEV
        elif metric == "SR":
            self.__class__ = OptimSR
        elif metric == "IR":
            self.__class__ = OptimIR
        elif metric == "Sortino":
            self.__class__ = OptimSortino
        elif metric == "Vol":
            self.__class__ = OptimVol
        elif metric == "Return":
            self.__class__ = OptimReturn
        elif metric == "RetVsBmk":
            self.__class__ = OptimRetVsBmk
        elif metric == "Beta":
            self.__class__ = OptimBeta
        elif metric == "Beta=1":
            self.__class__ = OptimBeta1

        # Run Optimization
        opt_weights = self.optimize()
        super().__init__(prices, weights=opt_weights, V0=V0, max_leverage=max_leverage, method=method, benchmark_rate=0)

        # salva oggetto con entrambe le serie storiche
        self.ts = pd.concat([self.ptf_ts, self.bmk_ts[self.bmk_ts.index >= self.ptf_ts.index.min()]], axis=1)
        self.ts = pa.rebase_ts(self.ts)
        #### Fin __init__()

    def check_rebal_dates(self):
        # check che tutte le rebalancing dates siano presenti in self.prices.index
        rebal_dates = self.rebal_dates
        if not set(rebal_dates).issubset(self.prices.index):
            # inserisci come rebal_date la data immediatamente precedente presente in self.prices.index
            for missing_rebal in set(rebal_dates).difference(self.prices.index):
                previous_date = min(self.prices.index[self.prices.index <= missing_rebal])
                rebal_dates.pop(rebal_dates.index(missing_rebal))
                rebal_dates.append(previous_date)
            rebal_dates = sorted(rebal_dates)
        # controlla che la prima data disponibile per gli asset sia maggiore di rebal_date - lookback in mesi
        if self.start_optim_date < self.prices.index[0]:
            first_rebal_date = [x for x in rebal_dates
                                if x > self.prices.index[0] + pd.tseries.offsets.DateOffset(months=self.lookback)]
            first_rebal_date = min(first_rebal_date)
            print(f"Removing rebalancing dates before {first_rebal_date.strftime('%Y-%m-%d')}")
            # ridefinire qui la data di inizio!
            rebal_dates = [x for x in rebal_dates if x > first_rebal_date]
            self.start_optim_date = min(rebal_dates) - pd.tseries.offsets.DateOffset(months=self.lookback)

        return rebal_dates

    def optimize_dd(self, rebal_date):
        start_date = rebal_date - pd.tseries.offsets.DateOffset(months=self.lookback)
        # subset on dates you need for the optimization
        prices = self.prices[(self.prices.index >= start_date) & (self.prices.index <= rebal_date)]
        bmk_ret = self.bmk_ret[(self.bmk_ret.index >= start_date) & (self.bmk_ret.index <= rebal_date)]
        bmk_ret.iloc[0] = 0
        # preparazione per ottimizzazione
        basket = Portfolio(prices)
        # Initial guess
        x0 = np.repeat(1 / len(prices.columns), len(prices.columns))
        # Constraints
        # Sum to 1
        sum_to_one = LinearConstraint(A=np.ones_like(x0), lb=1., ub=1.)
        # Bounds
        bounds = [(0., 1.) for _ in range(len(prices.columns))]
        # Launch optimizer
        res = minimize(
            fun=self.optim_fun,
            x0=x0,
            args=(bmk_ret, basket),
            method="SLSQP",
            bounds=bounds,
            constraints=sum_to_one,
            options={"ftol": 1e-7, "maxiter": 100, "disp": False}
        )
        if self.metric in ["SR", "IR", "Sortino", "Return", "Beta"]:
            init_metric = -self.optim_fun(x0, bmk_ret, basket) * 100
            opt_metric = -self.optim_fun(res.x, bmk_ret, basket) * 100
        else:
            init_metric = self.optim_fun(x0, bmk_ret, basket) * 100
            opt_metric = self.optim_fun(res.x, bmk_ret, basket) * 100
        print(f"Optimizing for {rebal_date.strftime('%Y-%m-%d')}. "
              f"Initial {self.metric}: {init_metric:.2f}%. "
              f"Optimized {self.metric}: {opt_metric:.2f}%.")
        weights = basket.prep_weights(res.x, rebal_date=rebal_date)
        return weights

    def optimize(self):
        weights = list()
        for rebald in self.rebal_dates:
            # print(f"Optimizing for rebalancing date {rebald.strftime('%Y-%m-%d')}")
            w_rd = self.optimize_dd(rebald)
            weights.append(w_rd)
        weights = pd.concat(weights, axis=0)
        return weights

    def plot_weights(self):
        self.weights.plot.area(stacked=True)

    def plot_perf(self):
        self.ts.plot()


class OptimTEV(PortfolioOptimization):
    def optim_fun(self, x, b, basket):
        weights = basket.prep_weights(x)
        r, _ = basket.portfolio_returns(weights=weights)
        tev = pa.compute_tracking_error_vol(r, b)
        return tev


class OptimSR(PortfolioOptimization):
    def optim_fun(self, x, b, basket):
        weights = basket.prep_weights(x)
        r, _ = basket.portfolio_returns(weights=weights)
        sr = pa.compute_sharpe_ratio(r, benchmark_rate=0)
        return -sr


class OptimIR(PortfolioOptimization):
    def optim_fun(self, x, b, basket):
        weights = basket.prep_weights(x)
        r, _ = basket.portfolio_returns(weights=weights)
        ir = pa.compute_information_ratio(r, b)
        return -ir


class OptimSortino(PortfolioOptimization):
    def optim_fun(self, x, b, basket):
        weights = basket.prep_weights(x)
        r, _ = basket.portfolio_returns(weights=weights)
        sortino = pa.compute_sortino_ratio(pa.compute_ts(r), benchmark_rate=0)
        return -sortino


class OptimVol(PortfolioOptimization):
    def optim_fun(self, x, b, basket):
        weights = basket.prep_weights(x)
        r, _ = basket.portfolio_returns(weights=weights)
        vol = pa.compute_annualized_volatility(r)
        return vol


class OptimReturn(PortfolioOptimization):
    def optim_fun(self, x, b, basket):
        weights = basket.prep_weights(x)
        r, _ = basket.portfolio_returns(weights=weights)
        cagr = pa.compute_cagr(pa.compute_ts(r))
        return -cagr


class OptimRetVsBmk(PortfolioOptimization):
    def optim_fun(self, x, b, basket):
        weights = basket.prep_weights(x)
        r, _ = basket.portfolio_returns(weights=weights)
        cagr = pa.compute_cagr(pa.compute_ts(r))
        cagr_bmk = pa.compute_cagr(pa.compute_ts(b))
        return abs(cagr - cagr_bmk)


class OptimBeta(PortfolioOptimization):
    def optim_fun(self, x, b, basket):
        weights = basket.prep_weights(x)
        r, _ = basket.portfolio_returns(weights=weights)
        beta = pa.compute_beta(r, b)
        return -beta


class OptimBeta1(PortfolioOptimization):
    def optim_fun(self, x, b, basket):
        weights = basket.prep_weights(x)
        r, _ = basket.portfolio_returns(weights=weights)
        beta = pa.compute_beta(r, b)
        return abs(beta - 1)
