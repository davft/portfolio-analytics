import pandas as pd
import numpy as np
from scipy.optimize import root_scalar


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
    if not isinstance(id_col, (str, list)):
        raise TypeError("id_col must be str or list of str")
    if isinstance(id_col, list):
        if not all(isinstance(item, str) for item in id_col):
            raise TypeError("all elements of id_col must be str")
    else:
        id_col = [id_col]
    if not isinstance(w_col, str):
        raise TypeError("w_col must be str")
    if not isinstance(keep_all_cols, bool):
        raise TypeError("keep_all_cols must be boolean")

    # check if all needed columns are in holdings
    required_cols = [*id_col, w_col, *groupby]
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
        return df.iloc[:cut_idx]

    if rebase is not None:
        # rebase weights
        holdings[w_col] = holdings[w_col] / holdings[w_col].sum() * rebase

    if keep_all_cols:
        # mantieni da parte le colonne non necessarie, fai il merge in fondo
        other_cols = list(set(holdings.columns).difference(required_cols))
        other_cols = [*other_cols, *id_col]
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
        replica = replica.groupby(groupby).apply(lambda x: x.iloc[:min(int(x["n_stocks"].unique()), x.shape[0])])
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
        replica = replica.merge(df_other_cols, how="left", on=id_col)

    # rename w_col in "Weight_Index", rename "w_replica" to w_col (easier to use later in the script)
    replica = replica.rename(columns={w_col: "Weight_Original"})
    replica = replica.rename(columns={"w_replica": w_col})

    # check if weights respect bounds, if not rebase them
    # perform this operation on a groupby-level, in order to maintain the same overall exposure
    replica = replica.groupby(groupby) \
        .apply(lambda x: redistribuite_weights_given_bounds(df=x, minw=minw, maxw=maxw, w_col=w_col)) \
        .reset_index(drop=True)

    return replica


def compute_lux_replica(hldg, N=100, groupby="GICS Sector", agg=None, w_col="weight", id_col="ISIN",
                        equally_weighted=True, rebase=1., final_w=1., use_scipy=True):
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
    :param final_w: int/float or None. if None, do not rebase final weights. if not None, rebase final weights to final_w
    :param use_scipy: boolean, which method to use to find N_fittizio.
        if True uses scipy.optimize.root_scalar()
        if False uses function implemented in MySQL (while loop)
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
            raise TypeError("rebase must be None or int/float")

    if final_w is not None:
        if not isinstance(final_w, (int, float)):
            raise TypeError("final_w must be None or int/float")

    # columns required for computing replica portfolio
    required_cols = [id_col, w_col, *groupby]
    if not set(required_cols).issubset(hldg.columns):
        raise Exception(f"`hldg` must have columns: {id_col}, {w_col}, {groupby}")

    if rebase is not None:
        # rebase weights
        hldg[w_col] = hldg[w_col] / hldg[w_col].sum() * rebase

    if agg is None:
        # calcolare agg se non è fornito
        agg = hldg.groupby(groupby)[w_col].sum().reset_index(name="w_groupby")
    else:
        if not {*groupby, "w_groupby"}.issubset(agg.columns):
            raise Exception(f"`agg` must have columns: {groupby}, w_groupby")

    if use_scipy:
        # scipy.optimize.root_scalar()
        def find_N_fittizio(x, series, N):
            return sum([round(x * y, 0) for y in series]) - N

        N_fittizio = root_scalar(find_N_fittizio, args=(agg["w_groupby"], N), bracket=[0, N + 50], x0=N).root
    else:
        # while-loop to find the minimum integer which gives at least N stocks
        N_fittizio = N
        while sum([round(N_fittizio * x, 0) for x in agg["w_groupby"]]) < N:
            N_fittizio += 1

    # calcola il numero di stock che vanno selezionate per ogni combinazione groupby
    agg["n_stocks"] = [int(round(N_fittizio * x, 0)) for x in agg["w_groupby"]]

    # rimuovi le combinazioni con 0 stock
    agg = agg[agg["n_stocks"] != 0]

    # se c'è GOOGL US (US02079K3059) rimuovi GOOG US (US02079K1079)
    if "US02079K3059" in hldg[id_col].values:
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
        # appoggiati al numero di titoli effettivo
        tmp = replica.groupby(groupby).size().reset_index(name="real_n_stocks")
        agg = agg.merge(tmp, how="left", on=groupby)
        agg["w_stock"] = agg["w_groupby"] / agg["real_n_stocks"]
        # agg["w_stock"] = agg["w_groupby"] / agg["n_stocks"]

        # aggiungi colonna contente il peso di ogni combinazione groupby
        replica = replica.merge(agg[[*groupby, "w_stock"]], how="left", on=groupby)
        replica["w_replica"] = replica["w_stock"]
        # replica["w_replica"] = replica["w_replica"] / sum(replica["w_replica"])
    else:
        # aggiungi il peso della combinazione groupby
        replica = replica.merge(agg[[*groupby, "w_groupby"]], how="left", on=groupby)
        replica["w_replica"] = \
            replica.groupby(groupby).apply(lambda x: x[w_col] / x[w_col].sum() * x["w_groupby"]).values

    # ribasa peso replica a 1
    if final_w is not None:
        replica["w_replica"] = replica["w_replica"] / replica["w_replica"].sum() * final_w

    output_cols = [id_col, w_col, *groupby, "w_replica"]
    replica = replica[output_cols]

    return replica


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
    if minw > maxw:
        raise ValueError("minw must be lower than maxw")

    # check if bounds are respected
    if (min(df[w_col]) >= minw) & (max(df[w_col]) <= maxw):
        # if yes, no adjustment needed
        return df

    # check if operation is feasible
    if sum(df[w_col]) < len(df) * minw:
        raise ValueError(f"infeasible: sum(df[w_col]) must be >= len(df) * minw ({len(df) * minw:.2%}), "
                         f"it is {sum(df[w_col]):.2%}")
    if sum(df[w_col]) > len(df) * maxw:
        raise ValueError(f"infeasible: sum(df[w_col]) must be <= len(df) * maxw ({len(df) * maxw:.2%}), "
                         f"it is {sum(df[w_col]):.2%}")

    # isolate values not in bound (and on its limits): [0, minw], [maxw, 1]
    out_bound = df[(df[w_col] <= minw) | (df[w_col] >= maxw)]
    out_bound = out_bound.copy()
    while len(out_bound[(out_bound[w_col] < minw) | (out_bound[w_col] > maxw)]):
        # compute how much weight you need to redistribuite across obs in bounds
        # 1. weight needed to bring values below minw to minw (negative quantity)
        excess_w = sum(out_bound[out_bound[w_col] < minw][w_col]) - len(out_bound[out_bound[w_col] < minw]) * minw
        # 2. excess weight you get from bringing values above maxw to maxw (positive quantity)
        excess_w += sum(out_bound[out_bound[w_col] > maxw][w_col]) - len(out_bound[out_bound[w_col] > maxw]) * maxw
        # floor weights to minw and cap to maxw
        out_bound.loc[:, w_col] = out_bound[w_col].clip(minw, maxw)

        btw_bound = df[(df[w_col] > minw) & (df[w_col] < maxw)]
        btw_bound = btw_bound.copy()
        btw_bound[w_col] = btw_bound[w_col] + btw_bound[w_col] / sum(btw_bound[w_col]) * excess_w

        # override df with the new weights
        df = pd.concat([btw_bound, out_bound], axis=0)
        # isolate again values not in bound to repeat the cycle if needed
        out_bound = df[(df[w_col] <= minw) | (df[w_col] >= maxw)]

    return df


def compute_groupby_difference(hldg, bmk_hldg, groupby="GICS Sector", w_col="Weight", w_col_bmk=None):
    """
    compute difference between aggregation of ptf vs index
    :param hldg: pd.DataFrame with portfolio composition [only one ptf per call]
    :param bmk_hldg: pd.DataFrame with benchmark composition
    :param groupby: list of keys to groupby
    :param w_col: str, weight colname
    :param w_col_bmk: str or None. if None, w_col_bmk = w_col, otherwise use w_col_bmk for bmk_hldg and w_col for hldg
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
    if w_col_bmk is None:
        w_col_bmk = w_col
    else:
        if not isinstance(w_col_bmk, str):
            raise TypeError("w_col_bmk must be str")

    # # columns to lowercase
    # hldg.columns = hldg.columns.map(lambda x: x.lower())
    # bmk_hldg.columns = bmk_hldg.columns.map(lambda x: x.lower())
    # groupby = list(map(lambda x: x.lower(), groupby))

    # group by `groupby` and compute allocation weights
    def fun_exposure(x, col):
        return x[col].sum()

    ptf_exposure = hldg.groupby(groupby).apply(lambda x: fun_exposure(x, w_col)).reset_index(name="replica_exp")
    bmk_exposure = bmk_hldg.groupby(groupby).apply(lambda x: fun_exposure(x, w_col_bmk)).reset_index(name="bmk_exp")

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