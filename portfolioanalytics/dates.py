import pandas as pd
from pandas.tseries.frequencies import to_offset
from datetime import datetime, date
from dateutil.relativedelta import relativedelta


def dt(d):
    """
    transforms date object into datetime object
    :param d:
    :return:
    """

    if isinstance(d, datetime):
        return d
    else:
        return datetime(d.year, d.month, d.day, 0, 0)


def tdy(as_datetime=False, now=False, as_timestamp=False):
    """
    Returns today's date as either date or datetime object
    :param as_datetime:
    :param now:
    :param as_timestamp:
    :return:
    """

    if as_datetime:
        if now:
            return datetime.now()
        else:
            now = datetime.today()
            return datetime(now.year, now.month, now.day)
    elif as_timestamp:
        return pd.Timestamp(date.today())
    else:
        return date.today()


def ds(ddate=None, offset="-1M"):
    """
    Shifts a date object using an appropriate dateoffset object (or string, see below)
    * if you add B (es/ -1BM): Business (working days)
    * if you add S (es/ -1MS): Start of following period (it yields offset="-1M" + 1D)
    * you can use B and S together (es/ -3BQS)
    Alias           Description
    W-SUN           weekly frequency (sundays). Same as "W"
    W-MON           weekly frequency (mondays)
    W-TUE           weekly frequency (tuesdays)
    W-WED           weekly frequency (wednesdays)
    W-THU           weekly frequency (thursdays)
    W-FRI           weekly frequency (fridays)
    W-SAT           weekly frequency (saturdays)
    (B)M(S)         monthly frequency
    (B)Q(S)-DEC     quarterly frequency, year ends in December. Same as "Q"
    (B)Q(S)-JAN     quarterly frequency, year ends in January
    (B)Q(S)-FEB     quarterly frequency, year ends in February
    (B)Q(S)-MAR     quarterly frequency, year ends in March
    (B)Q(S)-APR     quarterly frequency, year ends in April
    (B)Q(S)-MAY     quarterly frequency, year ends in May
    (B)Q(S)-JUN     quarterly frequency, year ends in June
    (B)Q(S)-JUL     quarterly frequency, year ends in July
    (B)Q(S)-AUG     quarterly frequency, year ends in August
    (B)Q(S)-SEP     quarterly frequency, year ends in September
    (B)Q(S)-OCT     quarterly frequency, year ends in October
    (B)Q(S)-NOV     quarterly frequency, year ends in November
    :param ddate:
    :param offset:
    :return:
    """

    if ddate is None:
        ddate = tdy()

    if isinstance(offset, str):
        i = 0
        while offset[i] in list("-+0123456789") and i < len(offset):
            i += 1

        if i < len(offset):
            mult = int(offset[:i])
            offs = to_offset(offset[i:])
        else:
            mult = 1
            offs = to_offset(offset)
    else:
        mult = 1
        offs = offset

    if isinstance(ddate, datetime):
        return pd.to_datetime(ddate + mult * offs)
    elif isinstance(ddate, date):
        return (dt(ddate) + mult * offs).date()
    else:
        raise Exception("ddate must be date or datetime.")


def pbd(n=1, ddate=None):
    """
    Yields n previous business days
    :param n: integer, number of business day to subtract
    :param ddate:
    :return:
    """
    return ds(ddate=ddate, offset=f"-{n}B")


def fbd(n=1, ddate=None):
    """
    Yields n forward business days
    :param n: integer, number of business days to add
    :param ddate:
    :return:
    """
    return ds(ddate=ddate, offset=f"+{n}B")


def get_end_of_weeks(dates, days=None):
    """

    :param dates: list or pandas.core.indexes.base.Index containing dates
    :param days: None, int or list of int
        if not None, then analysis is performed only on the give days (mon=0, sun=6)
    :return: end of weeks
    """
    if isinstance(dates, pd.core.indexes.datetimes.DatetimeIndex):
        dates = dates.values.tolist()

    dates = pd.to_datetime(dates)
    df = pd.DataFrame(dates, index=dates, columns=["Date"])
    if days is not None:
        if isinstance(days, (int, float)):
            days = [int(days)]
        else:
            days = [int(m) for m in days]
        # subset dates only on given days
        df = df[df.index.dayofweek.isin(days)]
    # x.diff(-1) = x.shift(-1) - x
    # val != 0 l'ultimo giorno prima del cambio settimana
    df["week_change"] = df.index.isocalendar().week.diff(-1).fillna(1)
    # mantieni solo le righe !=0
    df = df[df["week_change"] != 0]
    eow = pd.to_datetime(df["Date"].values.tolist())

    return eow


def get_end_of_months(dates, months=None):
    """

    :param dates: list or pandas.core.indexes.base.Index containing dates
    :param months: None, int or list of int
        if not None, then analysis is performed only on the give months (jan=1, dec=12)
    :return: end of months
    """

    if isinstance(dates, pd.core.indexes.datetimes.DatetimeIndex):
        dates = dates.values.tolist()

    dates = pd.to_datetime(dates)
    df = pd.DataFrame(dates, index=dates, columns=["Date"])
    if months is not None:
        if isinstance(months, (int, float)):
            months = [int(months)]
        else:
            months = [int(m) for m in months]
        # subset dates only on given months
        df = df[df.index.month.isin(months)]
    # group by year, month
    g = df.groupby([df.index.year, df.index.month])
    # get end of month
    eom = g.max()
    # eom["Date"] = [d.date() for d in eom["Date"]]
    # eom = eom["Date"]
    eom = pd.to_datetime(eom["Date"].values.tolist())

    return eom


def get_end_of_years(dates):
    """

    :param dates: list or pandas.core.indexes.base.Index containing dates
    :return: end of years
    """

    if isinstance(dates, pd.core.indexes.datetimes.DatetimeIndex):
        dates = dates.values.tolist()

    dates = pd.to_datetime(dates)
    df = pd.DataFrame(dates, index=dates, columns=["Date"])
    # group by year
    g = df.groupby(df.index.year)
    # get end of month
    eoy = g.max()
    eoy = pd.to_datetime(eoy["Date"].values.tolist())

    return eoy


# Dates convention
def dc_act360(start_date, end_date=None):
    """
    Computes the difference between two dates with act/360 day count convention
    :param start_date:
    :param end_date: in None, today
    :return:
    """
    if end_date is None:
        end_date = tdy()

    return (end_date.toordinal() - start_date.toordinal()) / 360.


def dc_act365(start_date, end_date=None):
    """
    Computes the difference between two dates with act/365 day count convention
    :param start_date:
    :param end_date: if None, today
    :return:
    """
    if end_date is None:
        end_date = tdy()

    return (end_date.toordinal() - start_date.toordinal) / 365.


def dc_30e360(start_date, end_date=None):
    """
    Computes the difference between two dates with 30E360 day count convention
    :param start_date:
    :param end_date: if None, today
    :return:
    """
    if end_date is None:
        end_date = tdy()

    d1, d2 = start_date.day, end_date.day
    m1, m2 = start_date.month, end_date.month
    y1, y2 = start_date.year, end_date.year

    if d1 == 31: d1 = 30
    if d2 == 31: d2 = 30

    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.


def dc_30e365(start_date, end_date=None):
    """
    Computes the difference between two dates with 30E365 day count convention
    :param start_date:
    :param end_date: if None, today
    :return:
    """
    if end_date is None:
        end_date = tdy()

    d1, d2 = start_date.day, end_date.day
    m1, m2 = start_date.month, end_date.month
    y1, y2 = start_date.year, end_date.year

    if d1 == 31: d1 = 30
    if d2 == 31: d2 = 30

    return (365 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 365.


