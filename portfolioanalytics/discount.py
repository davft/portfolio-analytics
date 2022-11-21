import numpy_financial as npf


def compute_irr(T=10, S0=1000, monthly_s=100, F=20000, verbose=False):
    """
    compute required IRR to match investment expectation
    :param T: int or float, in 1/12 of month (es 10.5 for 10y6m). Time period in year
    :param S0: initial invested amount
    :param monthly_s: monthly investment
    :param F: final amount
    :return: required IRR to meet investment expectation
    """

    # number of periods
    M = 12  # frequency, monthly
    n_periods = round(T * M) - 1

    IRR = npf.irr([-(S0 + monthly_s), *[-monthly_s for _ in range(n_periods - 1)], F])
    IRR_yearly = (1 + IRR) ** M - 1

    if verbose:
        print(f"Total investment: {S0 + n_periods * monthly_s:,.2f}, "
              f"required annual rate of return to get {F:,.2f}: {IRR_yearly:.2%}")

    return IRR_yearly


def compute_monthly_investment(T=10, S0=0, IRR=.1, F=10000, verbose=False):
    """
    compute monthly investment to match investment expectation
    :param T: int or float, in 1/12 of month (es 10.5 for 10y6m). Time period in year
    :param S0: initial invested amount
    :param IRR: internal rate of return of the investment (YEARLY!!)
    :param F: final amount
    :return: required monthly investment to meet investment expectation
    """
    # number of period, frequency monthly
    M = 12
    n_periods = round(T * M) - 1

    IRR_m = (1 + IRR) ** (1 / M) - 1

    discount = sum([(1 + IRR_m) ** i for i in range(1, n_periods + 1)])

    monthly_s = (F - S0 * (1 + IRR_m) ** n_periods) / discount

    if verbose:
        print(f"The monthly investment required to get {F} in {T} years with an initial investment of {S0} is "
              f"{monthly_s}")

    return monthly_s
