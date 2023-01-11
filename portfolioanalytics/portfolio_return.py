# https://cran.r-project.org/web/packages/PerformanceAnalytics/vignettes/portfolio_returns.pdf
import pandas as pd
import numpy as np
import warnings
# https://stackoverflow.com/questions/16004076/python-importing-a-module-that-imports-a-module
from . import functions as pa


class Portfolio(object):
    """

    """

    def __init__(self, prices, weights=None, V0=100, max_leverage=1, method="simple", benchmark_rate=0):
        """

        :param prices: pd.DataFrame with price time series in columns
        :param weights: None or pd.DataFrame with weights. if None, then buy-and-hold equally weighted
        :param V0: initial portfolio value
        :param max_leverage: float, the maximum investment. if sum(w) > max_leverage, then rebase to max_leverage.
                            if sum(w) < max_leverage, then create residual weight with zero returns.
                            if None, do not adjust weights.
        :param method: simple or log returns
        :param benchmark_rate: annualized benchmark rate, defaults to 0
        """

        assert isinstance(prices, pd.DataFrame)
        self.method = method
        self.prices = prices.copy()
        self.max_leverage = max_leverage
        if weights is None:
            self.weights = self.set_bh_ew_weights()
        else:
            self.weights = self.check_weights(weights.copy())
        self.V0 = V0
        self.ptf_ret, self.ptf_ts = self.portfolio_returns()
        self.benchmark_rate = benchmark_rate

    def __repr__(self):
        return f"Portfolio() with {len(self.weights.columns)} assets"
    
    def __str__(self):
        # usage: print(object)
        ann_ret = pa.compute_cagr(self.ptf_ts)
        ann_std = pa.compute_annualized_volatility(self.ptf_ret)
        SR = pa.compute_sharpe_ratio(self.ptf_ret, self.benchmark_rate)
        DD = pa.compute_max_drawdown(self.ptf_ts, self.method)
        VaR = pa.compute_historical_var(self.ptf_ret, conf_lvl=.95)
        return f"""
        {'-' * 45}
        * Portfolio with {len(self.weights.columns)} assets
        * Analysed period: {min(self.prices.index).strftime('%Y-%m-%d')} - {max(self.prices.index).strftime('%Y-%m-%d')}
        * Number of rebalancing events:  {len(self.weights)}
        * Annualized Return:             {ann_ret:.2%}
        * Annualized Standard Deviation: {ann_std:.2%}
        * Sharpe Ratio:                  {SR:.2%}
        * Max Drawdown:                  {DD:.2%}
        * Historical VaR 95%:            {VaR:.2%}
        {'-' * 45}
        """

    def set_bh_ew_weights(self):
        """
        returns a pd.DataFrame with weights of buy-and-hold equally weighted ptf
        """
        N = self.prices.shape[1]
        weights = pd.DataFrame([np.repeat(1 / N, N)], index=[self.prices.index[0]], columns=self.prices.columns)
        return weights

    def check_weights(self, weights):

        if isinstance(weights, (list, np.ndarray)):
            # assume buy & hold portfolio with specified weights
            weights = pd.DataFrame([weights], index=[self.prices.index[0]], columns=self.prices.columns)

        if self.max_leverage is not None:
            tol = 1e-06
            if any(weights.sum(axis=1) > self.max_leverage + tol):
                warnings.warn("\nsum of weights exceed max_leverage value of {} in dates {}:\nrebasing to {}".format(
                    self.max_leverage, weights[weights.sum(axis=1) > self.max_leverage].index.values, self.max_leverage))
                # ribasa i pesi per le date in cui superano max_leverage + tolleranza
                weights[weights.sum(axis=1) > self.max_leverage] = \
                    weights[weights.sum(axis=1) > self.max_leverage].apply(
                        lambda x: x / sum(x) * self.max_leverage, axis=1
                    )

        if not all(np.isclose(weights.sum(axis=1), 1, rtol=1e-06)):
            warnings.warn(
                "\none or more rebalancing dates have weights not summing up to 100%:\n" +
                "adding a residual weight to compensate")
            weights["residual"] = 1 - weights.sum(axis=1)

        return weights

    def prep_weights(self, x, rebal_date=None):
        """
        ausiliar function, to prepare weights
        :param x: list or np.ndarray
        :return:
        """
        weights = pd.DataFrame(x).transpose()
        weights.columns = self.prices.columns
        if rebal_date is None:
            weights.index = [self.prices.index[0]]
        else:
            weights.index = [rebal_date]

        return weights

    def get_components_value_single_period(self, ret, v_init):
        """
        compute components values over time, in a single rebalancing window, given returns and initial values
        :param ret: pd.DataFrame, with .index dates and containing components returns over time
        :param v_init: initial components values
        :return:
        """

        if isinstance(v_init, pd.Series):
            v_init = [v_init.values.tolist()]
        elif isinstance(v_init, pd.DataFrame):
            v_init = v_init.values.tolist()
        else:
            raise ValueError("v_init should be either pd.Series or pd.DataFrame")
        
        components_value = pd.DataFrame(v_init * len(ret), index=ret.index, columns=ret.columns)
        if self.method == "simple":
            components_value = components_value * ret.apply(lambda x: np.cumprod(1 + x), axis=0)
        elif self.method == "log":
            components_value = components_value * ret.apply(lambda x: np.cumsum(x), axis=0)
        else:
            raise ValueError("method should be either simple or log")

        return components_value

    def portfolio_returns(self, weights=None, V0=None, max_leverage=None, verbose=False):
        """

        :param weights: if None use self.weights,
                            otherwise use given input and update self.weights, self.ptf_ret, self.ptf_ts
        :param V0: if None use self.V0, else float, initial portfolio value, overrides self.V0
        :param max_leverage: if None use self.max_leverage, else use max_leverage.
            it is not possible to change max_leverage from a number to None, in that case you need to reinitiate the class
        :param verbose: if True, returns components contributions to portfolio returns
        :return: portfolio returns. if verbose=True, return tuple with ptf rets, contribs
        """

        # update inputs
        if V0 is None:
            V0 = self.V0
        else:
            self.V0 = V0

        if max_leverage is None:
            max_leverage = self.max_leverage
        else:
            self.max_leverage = max_leverage

        # compute stocks returns
        returns = pa.compute_returns(self.prices, method=self.method)

        if weights is None:
            # potrebbe esserci stato l'update di max_leverage
            weights = self.check_weights(self.weights)
        else:
            weights = self.check_weights(weights.copy())
            assert isinstance(weights, pd.DataFrame)
            self.weights = weights

        if "residual" in weights.columns:
            returns["residual"] = 0

        # subset returns to match weights.columns
        returns = returns[weights.columns.tolist()]
        # subset weights to be inside returns dates
        idx = [ind for ind in weights.index if ind in returns.index[:-1]]
        if idx != weights.index.to_list():
            warnings.warn("Some rebalancing dates don't match prices dates. Non matching dates will not be considered.")
            weights = weights.loc[idx]

        V_bop = list()
        V = list()
        n_iter = len(weights.index)

        for t in range(n_iter):
            if t == 0:  # first rebalancing date,
                # get the values of each component at first rebalancing date
                v_bop = V0 * weights.iloc[t]
            else:
                # not the first rebal date, set v_init equal to last available V
                v_bop = V[-1].tail(1).sum(axis=1).values * weights.iloc[t]

            V_bop.append(v_bop.to_frame().transpose())

            # subset returns
            if t != n_iter - 1:
                tmp_ret = returns.loc[weights.index[t]:weights.index[t + 1]]
            else:
                # se è l'ultima iterazione prendi i ritorni fino all'ultima data disponibile
                tmp_ret = returns.loc[weights.index[t]:]

            # notice that subsetting by index includes both extremes!
            # we need to remove the first return, since rebalancing happens from the day after
            # the actual index indicated in the weights input
            tmp_ret = tmp_ret.drop(index=weights.index[t])
            # metti i ritorni mancanti a zero, altrimenti ci sono "buchi" nel ptf.
            # esempio: se V0 = 100 e w1 = 10%, ma la stock 1 non ha ritorni in quel periodo, il ptf in t0 sommerà a 90 e
            # non a 100
            tmp_ret = tmp_ret.fillna(0)
            # cumulate returns components inside this interval, i.e. in
            # (index[t] + 1, index[t+1]]
            tmp_value = self.get_components_value_single_period(tmp_ret, v_bop)
            # append values both to V_bop and to V
            # to V_bop we attach not the last value, since the last bop will
            # be replaced by the new v_bop
            V_bop.append(tmp_value.iloc[:-1])
            V.append(tmp_value)

        # concat results to get the full components values over time

        # we attach to V the first element
        # corresponding to the first V_bop,
        # notice that this is a bit fictitious, since
        # the eop of the very first rebalancing day is not known,
        # we only know the bop of the day after the rebalancing day
        V.insert(0, V_bop[0])
        V = pd.concat(V)
        # here we need to attach an even more fictitious term,
        # the bop of the first rebalancing day,
        # this is done only for index compatibility with V, it does not matter
        V_bop.insert(0, V_bop[0])
        V_bop = pd.concat(V_bop)
        # assign index to values, index starts at the first date of rebalancing
        V.index = returns.loc[weights.index[0]:].index
        V_bop.index = returns.loc[weights.index[0]:].index

        # portfolio timeseries
        ptf = V.sum(axis=1)
        # portfolio returns
        ptf_ret = pa.compute_returns(ptf, method=self.method)

        self.ptf_ret = ptf_ret
        self.ptf_ts = ptf

        if verbose:
            # compute components' contributions in each day via
            # contrib_i = V_i - Vbop_i / sum(Vbop)
            contrib = V.add(-V_bop).divide(V_bop.sum(axis=1), axis=0)
            # check if sum di contrib = ptf_ret
            # np.sum(np.abs(contrib.apply(sum, axis=1).subtract(ptf_ret)))

            # calcola il turnover
            turnover = V_bop.shift(-1).subtract(V)
            turnover = turnover.loc[weights.index]
            # old approach: divided by EOP ptf value
            # turnover = turnover.apply(lambda x: np.sum(np.abs(x)), axis=1).divide(ptf.loc[weights.index])
            # new approach: divided by average ptf value
            # denominator: average portfolio value btw rebalancing dates
            avg_ptf = self.get_avg_ptf_value(ptf, weights)
            turnover = turnover.apply(lambda x: np.sum(np.abs(x)), axis=1).divide(avg_ptf)
            # secondo la definizione di cui sopra, il massimo turnover è 2. se modifichiamo l'allocazione dell'intero
            # ptf vogliamo turnover=1
            # turnover = turnover / 2
            return ptf_ret, ptf, contrib, turnover, V, V_bop

        return ptf_ret, ptf
    
    def get_eop_weights(self, weights=None):
        """
        
        :param weights: 
        :return: 
        """
        _, ts, _, _, V_eop, _ = self.portfolio_returns(weights=weights, verbose=True)
        # compute end of period weights dividing the end of period value of each stock by the eop ptf value
        w_eop = V_eop.divide(ts, axis=0)
        # seleziona i pesi end of period ad ogni chiusura prima del ribilanciamento
        dates = [*self.weights, w_eop.index[-1]]
        # drop first date: è la prima data di rebalancing
        del dates[0]
        w_eop = w_eop.loc[dates]

        return w_eop
    
    def get_last_eop_weights(self, weights=None):
        """
        
        :param weights: 
        :return: 
        """
        w_eop = self.get_eop_weights(weights=weights)
        # get last w_eop: il peso delle stock all'ultima data di prices
        # https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
        last_w_eop = w_eop.iloc[[-1]]

        return last_w_eop

    def get_avg_ptf_value(self, ptf, weights):
        """
        Computes average portfolio value between each rebalancing date.
        Output used for computing portfolio turnover (denominator)
        :param ptf: pd.Series containing ptf values
        :param weights: pd.DataFrame with index = rebalancing dates
        :return:
        """
        rebald = weights.index

        avg_ptf = list()
        for i in range(len(rebald)):
            if i == 0:
                avg_ptf.append(pd.Series(self.V0, index=[rebald[i]]))
            else:
                tmp = ptf.loc[(ptf.index > rebald[i - 1]) & (ptf.index <= rebald[i])]
                avg_ptf.append(pd.Series(np.mean(tmp), index=[rebald[i]]))

        avg_ptf = pd.concat(avg_ptf, axis=0)
        return avg_ptf
        
        

# #
# # prices = {
# #     "A": [10, 11, 13, 12, 12, 13, 15],
# #     "B": [20, 21, 19, 18, 19, 17, 15],
# #     "C": [20, 20, 21, 22, 23, 25, 24]
# # }
# # prices = pd.DataFrame(prices)
# # prices.index = [dt.date.today() - dt.timedelta(days=x) for x in range(prices.shape[0])][::-1]
# #
# # weights = {
# #     "A": [.5, .4],
# #     "B": [.3, .3],
# #     "C": [.2, .3]
# # }
# # weights = pd.DataFrame(weights)
# # weights.index = prices.index[0::4]
# #
# prova = Portfolio(prices=prices, weights=weights)
# prova.ptf_ts
#
# returns = pa.compute_returns(prices)
# V0 = 100
# max_leverage = 1
#
# V_bop = list()
# V = list()
# n_iter = len(weights.index)
#
# for t in range(n_iter):
#     if t == 0:  # first rebalancing date,
#         # get the values of each component at first rebalancing date
#         v_bop = V0 * weights.iloc[t]
#     else:
#         # not the first rebal date, set v_init equal to last available V
#         v_bop = V[-1].tail(1).sum(axis=1).values * weights.iloc[t]
#
#     V_bop.append(v_bop.to_frame().transpose())
#
#     # subset returns
#     if t != n_iter - 1:
#         tmp_ret = returns.loc[weights.index[t]:weights.index[t + 1]]
#     else:
#         # se è l'ultima iterazione prendi i ritorni fino all'ultima data disponibile
#         tmp_ret = returns.loc[weights.index[t]:]
#
#     # notice that subsetting by index includes both extremes!
#     # we need to remove the first return, since rebalancing happens from the day after
#     # the actual index indicated in the weights input
#     tmp_ret = tmp_ret.drop(index=weights.index[t])
#     # cumulate returns components inside this interval, i.e. in
#     # (index[t] + 1, index[t+1]]
#     tmp_value = prova.get_components_value_single_period(tmp_ret, v_bop)
#     # append values both to V_bop and to V
#     # to V_bop we attach not the last value, since the last bop will
#     # be replaced by the new v_bop
#     V_bop.append(tmp_value.iloc[:-1])
#     V.append(tmp_value)
#
# # concat results to get the full components values over time
#
# # we attach to V the first element
# # corresponding to the first V_bop,
# # notice that this is a bit fictitious, since
# # the eop of the very first rebalancing day is not known,
# # we only know the bop of the day after the rebalancing day
# V.insert(0, V_bop[0])
# V = pd.concat(V)
# # here we need to attach an even more fictitious term,
# # the bop of the first rebalancing day,
# # this is done only for index compatibility with V, it does not matter
# V_bop.insert(0, V_bop[0])
# V_bop = pd.concat(V_bop)
# # assign index to values, index starts at the first date of rebalancing
# V.index = returns.loc[weights.index[0]:].index
# V_bop.index = returns.loc[weights.index[0]:].index
#
# # portfolio timeseries
# ptf = V.sum(axis=1)
# # portfolio returns
# ptf_ret = pa.compute_returns(ptf)
#
#
# # calcola il turnover
# turnover = V_bop.shift(-1).subtract(V)
# turnover = turnover.loc[weights.index]
# turnover = turnover.apply(lambda x: np.sum(np.abs(x)), axis=1).divide(ptf.loc[weights.index])
# # secondo la definizione di cui sopra, il massimo turnover è 2. se modifichiamo l'allocazione dell'intero
# # ptf vogliamo turnover=1
# turnover = turnover / 2
#
# # compute components' contributions in each day via
# # contrib_i = V_i - Vbop_i / sum(Vbop)
# contrib = V.add(-V_bop).divide(V_bop.sum(axis=1), axis=0)
# # check if sum di contrib = ptf_ret
# # np.sum(np.abs(contrib.apply(sum, axis=1).subtract(ptf_ret)))
#
#

