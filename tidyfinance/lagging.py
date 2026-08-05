"""Lagging and rolling-window functions for tidyfinance."""

import datetime as dt

import numpy as np
import polars as pl

from ._internal import _check_new_col, _negate_offset, _offset_end, _to_offset
from .backend import get_backend


def add_lagged_columns(
    data: pl.DataFrame,
    cols: list[str] | str,
    lag,
    max_lag=None,
    by: list[str] | str | None = None,
    drop_na: bool = False,
    ff_adjustment: bool = False,
    date_col: str = "date",
    data_options: dict | None = None,
) -> pl.DataFrame:
    """Append lagged columns to a data frame via a join-based approach.

    When 'lag == max_lag' (the default), an equi-join is used: source
    dates are shifted forward by 'lag' and matched exactly. When
    'lag < max_lag', an inequality join is used: for each row, the most
    recent source value within the window '[date - max_lag, date - lag]'
    is selected.

    The combination of 'by' and the date column must be unique in 'data'.
    If 'by' is None, dates alone must be unique.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the variables to lag. The date column must
        be of dtype 'pl.Date' or 'pl.Datetime'.
    cols : list of str or str
        Names of the columns to lag. Each column produces a new column
        suffixed with '_lag'.
    lag : int, str, datetime.timedelta, or pd.DateOffset
        Minimum lag (inclusive) to apply, e.g. '"1mo"'. An int is
        interpreted as days; strings are polars offset strings.
    max_lag : int, str, datetime.timedelta, or pd.DateOffset, optional
        Maximum lag (inclusive) to apply. Defaults to 'lag' (exact lag).
    by : list of str or str, optional
        Grouping column(s) (e.g. a stock identifier). Lagged values are
        matched within groups. Defaults to None.
    drop_na : bool, optional
        If True, missing values in the source columns are excluded
        before matching, so the lookup skips over missing observations.
        Applied independently per column. Defaults to False.
    ff_adjustment : bool, optional
        If True, only the last observation per year (within each group
        defined by 'by') is retained as a source for lagged values,
        following Fama-French conventions for annual accounting data.
        Defaults to False.
    date_col : str, optional
        Name of the date column. Defaults to 'date'.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'date' element is
        used to specify the date column. Uses the 'data_options' default
        when None: 'date' -> 'date'.

    Returns
    -------
    pl.DataFrame
        Data frame with the same rows as 'data' and new columns
        appended, each suffixed with '_lag'. Unmatched rows receive
        null in the lagged columns.

    Examples
    --------
    ```python
    import numpy as np
    import polars as pl
    import datetime as dt
    from tidyfinance import add_lagged_columns
    rng = np.random.default_rng(42)
    dates = pl.date_range(
        dt.date(2023, 1, 1), dt.date(2023, 10, 1), "1mo", eager=True
    )
    data = pl.DataFrame({
        'permno': [1] * 10 + [2] * 10,
        'date': dates.to_list() * 2,
        'size': rng.uniform(100, 200, 20),
        'bm': rng.uniform(0.5, 1.5, 20),
    })
    # Exact lag: each row gets the value from exactly 2 months earlier
    add_lagged_columns(data, cols=['size', 'bm'], lag='2mo', by='permno')
    # Window lag: most recent value from 2 to 4 months earlier
    add_lagged_columns(
        data, cols='size', lag='2mo', max_lag='4mo', by='permno'
    )
    ```
    """
    if data_options is not None:
        date_col = data_options.get("date", date_col)

    if isinstance(cols, str):
        cols = [cols]
    if isinstance(by, str):
        by = [by]
    by_list = by or []

    lag_offset = _to_offset(lag)
    max_lag_offset = _to_offset(max_lag if max_lag is not None else lag)

    if date_col not in data.columns:
        raise ValueError(f"'data' must contain the date column '{date_col}'.")

    ref = dt.date(2020, 1, 1)
    lag_end = _offset_end(ref, lag_offset)
    max_lag_end = _offset_end(ref, max_lag_offset)
    if lag_end < ref or max_lag_end < lag_end:
        raise ValueError(
            "'lag' and 'max_lag' must be non-negative and 'max_lag' "
            "must be >= 'lag'."
        )

    missing_cols = [c for c in cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"'data' is missing column(s): {missing_cols}.")

    if by_list:
        missing_by = [c for c in by_list if c not in data.columns]
        if missing_by:
            raise ValueError(
                f"'data' is missing grouping column(s): {missing_by}."
            )

    join_cols = by_list + [date_col]
    if data.select(join_cols).is_duplicated().any():
        raise ValueError(
            "The combination of 'by' and date columns must be unique in 'data'."
        )

    exact_lag = lag_end == max_lag_end
    result = data

    if not exact_lag:
        _check_new_col(result, ["_upper", "_lower"])
        result = result.with_columns(
            pl.col(date_col)
            .dt.offset_by(_negate_offset(lag_offset))
            .alias("_upper"),
            pl.col(date_col)
            .dt.offset_by(_negate_offset(max_lag_offset))
            .alias("_lower"),
        )

    for col in cols:
        lag_col_name = f"{col}_lag"
        if lag_col_name in result.columns:
            raise ValueError(
                f"Column '{lag_col_name}' already exists in 'data'."
            )

        lagged = data.select(join_cols + [col])

        if drop_na:
            keep = pl.col(col).is_not_null()
            if lagged.schema[col].is_float():
                keep = keep & pl.col(col).is_not_nan()
            lagged = lagged.filter(keep)

        if ff_adjustment:
            grp = by_list + [pl.col(date_col).dt.year()]
            lagged = lagged.filter(
                pl.col(date_col) == pl.col(date_col).max().over(grp)
            )

        if exact_lag:
            lagged = lagged.with_columns(
                pl.col(date_col).dt.offset_by(lag_offset)
            ).rename({col: lag_col_name})
            result = result.join(
                lagged,
                on=join_cols,
                how="left",
                maintain_order="left",
                coalesce=True,
            )
        else:
            result = _window_lag_join(
                result, lagged, by_list, date_col, col, lag_col_name
            )

    if not exact_lag:
        result = result.drop(["_upper", "_lower"])

    return result


def _window_lag_join(
    result: pl.DataFrame,
    lagged: pl.DataFrame,
    by_list: list[str],
    date_col: str,
    col: str,
    lag_col_name: str,
) -> pl.DataFrame:
    """Backward window join used by add_lagged_columns for non-exact lags.

    For each row in result (which already carries _upper and _lower
    bounds from the caller), finds the most recent row in lagged whose
    date falls within the window [_lower, _upper] and copies its col
    value into a new column named lag_col_name. The match is performed
    by group when by_list is non-empty.

    Internally uses a backward join_asof on _upper to locate the
    closest source date at or before the upper bound, then filters out
    rows whose source date falls below the lower bound. The original
    row order of result is preserved.

    Parameters
    ----------
    result : pl.DataFrame
        Target frame. Must contain the columns in by_list plus
        _upper and _lower (window bounds). Must not contain a
        column named _orig_idx.
    lagged : pl.DataFrame
        Source frame. Must contain the columns in by_list plus
        date_col and col. Must not contain a column named _src_date.
    by_list : list of str
        Grouping columns shared by both frames. Pass an empty list
        for an ungrouped join.
    date_col : str
        Name of the date column in lagged.
    col : str
        Name of the source value column in lagged to copy.
    lag_col_name : str
        Name of the new column to add to result with the matched
        source values. Unmatched rows receive null.

    Returns
    -------
    pl.DataFrame
        result with lag_col_name appended, in the original row order.
        The helper columns _orig_idx and _src_date are removed before
        return; the caller is responsible for dropping _upper and
        _lower.
    """
    _check_new_col(result, "_orig_idx")
    _check_new_col(lagged, "_src_date")
    result = result.with_row_index("_orig_idx")
    lagged = lagged.rename({date_col: "_src_date", col: lag_col_name})

    left_sorted = result.sort("_upper", maintain_order=True)
    right_sorted = lagged.sort("_src_date", maintain_order=True)

    merged = left_sorted.join_asof(
        right_sorted,
        left_on="_upper",
        right_on="_src_date",
        by=by_list if by_list else None,
        strategy="backward",
        check_sortedness=False,
    )

    merged = merged.with_columns(
        pl.when(
            pl.col("_src_date").is_not_null()
            & (pl.col("_src_date") >= pl.col("_lower"))
        )
        .then(pl.col(lag_col_name))
        .otherwise(None)
        .alias(lag_col_name)
    )

    return merged.sort("_orig_idx", maintain_order=True).drop(
        ["_orig_idx", "_src_date"]
    )


def join_lagged_values(
    original_data: pl.DataFrame,
    new_data: pl.DataFrame,
    id_keys: list[str] | str,
    min_lag,
    max_lag,
    ff_adjustment: bool = False,
    date_col: str = "date",
    data_options: dict | None = None,
) -> pl.DataFrame:
    """Join lagged values of variables over a date range.

    Joins lagged values of selected variables from one data frame
    ('new_data') into another ('original_data'), based on date ranges
    defined by 'min_lag' and 'max_lag'. Unlike 'add_lagged_columns',
    this function supports joining across data frames with different
    date grids (e.g. monthly source data into quarterly target data).
    All columns in 'new_data' besides 'id_keys' and the date column are
    lagged and joined under their original names.

    Parameters
    ----------
    original_data : pl.DataFrame
        Target panel data. The date column must be of dtype 'pl.Date'
        or 'pl.Datetime'.
    new_data : pl.DataFrame
        Source variables to lag and merge. All columns besides
        'id_keys' and the date column will be lagged and joined.
    id_keys : list of str or str
        Identifier column(s) shared by both frames.
    min_lag : int, str, datetime.timedelta, or pd.DateOffset
        Lower lag bound (inclusive).
    max_lag : int, str, datetime.timedelta, or pd.DateOffset
        Upper lag bound (inclusive).
    ff_adjustment : bool, optional
        If True, keeps only the last observation per identifier and
        year in 'new_data' before lagging (Fama-French convention).
        Defaults to False.
    date_col : str, optional
        Name of the date column. Defaults to 'date'.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'date' element is
        used to identify the date column. Uses the 'data_options'
        default when None: 'date' -> 'date'.

    Returns
    -------
    pl.DataFrame
        'original_data' with all columns from 'new_data' appended as
        lagged values (keeping their original names).

    Examples
    --------
    ```python
    import numpy as np
    import polars as pl
    import datetime as dt
    from tidyfinance import join_lagged_values
    rng = np.random.default_rng(42)
    dates = pl.date_range(
        dt.date(2020, 1, 1), dt.date(2020, 6, 1), "1mo", eager=True
    )
    df1 = pl.DataFrame({
        'id': [1] * 6 + [2] * 6,
        'date': dates.to_list() * 2,
    })
    df2 = df1.with_columns(
        x=pl.Series(rng.standard_normal(len(df1)))
    )
    join_lagged_values(
        original_data=df1,
        new_data=df2,
        id_keys='id',
        min_lag='1mo',
        max_lag='3mo',
    )
    ```
    """
    if data_options is not None:
        date_col = data_options.get("date", date_col)

    if isinstance(id_keys, str):
        id_keys = [id_keys]
    if not isinstance(id_keys, list) or not all(
        isinstance(k, str) for k in id_keys
    ):
        raise ValueError("'id_keys' must be a string or list of strings.")

    min_lag_offset = _to_offset(min_lag)
    max_lag_offset = _to_offset(max_lag)

    if date_col not in original_data.columns:
        raise ValueError(
            f"'original_data' must contain the column '{date_col}'."
        )
    if date_col not in new_data.columns:
        raise ValueError(f"'new_data' must contain the column '{date_col}'.")

    missing_original = [k for k in id_keys if k not in original_data.columns]
    if missing_original:
        raise ValueError(
            f"'original_data' is missing id column(s): {missing_original}."
        )

    missing_new = [k for k in id_keys if k not in new_data.columns]
    if missing_new:
        raise ValueError(f"'new_data' is missing id column(s): {missing_new}.")

    new_column_names = [
        c for c in new_data.columns if c not in id_keys + [date_col]
    ]
    if not new_column_names:
        raise ValueError(
            f"'new_data' must contain columns besides {id_keys} and "
            f"'{date_col}'."
        )

    original_non_key = [
        c for c in original_data.columns if c not in id_keys + [date_col]
    ]
    duplicate_cols = [c for c in new_column_names if c in original_non_key]
    if duplicate_cols:
        raise ValueError(
            f"Column(s) in 'new_data' already exist in "
            f"'original_data': {duplicate_cols}. Remove or rename them "
            "before joining."
        )

    # Align date dtypes across the two frames so the asof join keys
    # match (e.g. Date vs Datetime, or differing time units).
    orig_dtype = original_data.schema[date_col]
    if new_data.schema[date_col] != orig_dtype:
        new_data = new_data.with_columns(pl.col(date_col).cast(orig_dtype))

    helper_cols = ["_lower", "_upper"]
    if ff_adjustment:
        helper_cols.append("_year")
    _check_new_col(new_data, helper_cols)
    new_data = new_data.with_columns(
        pl.col(date_col).dt.offset_by(min_lag_offset).alias("_lower"),
        pl.col(date_col).dt.offset_by(max_lag_offset).alias("_upper"),
    )

    result = original_data

    for col in new_column_names:
        select_cols = id_keys + [date_col, col, "_lower", "_upper"]
        tmp = new_data.select(select_cols)

        if ff_adjustment:
            grp = id_keys + [pl.col(date_col).dt.year()]
            tmp = tmp.filter(
                pl.col(date_col) == pl.col(date_col).max().over(grp)
            )
        tmp = tmp.drop(date_col)

        _check_new_col(result, "_orig_idx")
        result = result.with_row_index("_orig_idx")
        # join_asof requires the merge key sorted in ascending order;
        # the by= argument handles the grouping itself, so we only
        # sort by the merge key.
        left_sorted = result.sort(date_col, maintain_order=True)
        right_sorted = tmp.sort("_lower", maintain_order=True)

        merged = left_sorted.join_asof(
            right_sorted,
            left_on=date_col,
            right_on="_lower",
            by=id_keys if id_keys else None,
            strategy="backward",
            check_sortedness=False,
        )

        merged = merged.with_columns(
            pl.when(
                pl.col("_lower").is_not_null()
                & (pl.col(date_col) <= pl.col("_upper"))
            )
            .then(pl.col(col))
            .otherwise(None)
            .alias(col)
        )

        result = merged.sort("_orig_idx", maintain_order=True).drop(
            ["_orig_idx", "_lower", "_upper"]
        )

    return result


def compute_rolling_value(
    data: pl.DataFrame,
    f,
    period: str = "month",
    periods: int = 12,
    min_obs: int = None,
    data_options: dict = None,
) -> np.ndarray:
    """Compute a rolling value by period.

    Applies an arbitrary summary function over rolling time-period
    windows. Each window spans 'periods' units of 'period' (e.g., 12
    months). Before calling 'f', rows with any missing values are
    dropped from the window; if fewer than 'min_obs' rows remain, the
    result is NaN instead.

    Parameters
    ----------
    data : pl.DataFrame
        Data frame with a date column named according to
        'data_options[date]' (default 'date'). The column must be of
        dtype 'pl.Date' or 'pl.Datetime'.
    f : callable
        Function applied to each window. Receives the window slice
        (complete cases only) as a data frame of the active backend
        type ('pd.DataFrame' under the default 'pandas' backend,
        'pl.DataFrame' under the 'polars' backend) and must return
        a single scalar value.
    period : str, default 'month'
        Calendar period unit for the rolling windows. One of 'month',
        'quarter', or 'year'.
    periods : int, default 12
        Number of periods to include in the rolling window.
    min_obs : int, optional
        Minimum number of non-missing rows required per window.
        Defaults to 'periods'.
    data_options : dict, optional
        Column-name mapping (see 'data_options'). The 'date' element
        is used to specify the date column. Uses the 'data_options'
        default when None: 'date' -> 'date'.

    Returns
    -------
    np.ndarray
        Numeric vector aligned with the rows of 'data'.

    Examples
    --------
    ```python
    import numpy as np
    import polars as pl
    import datetime as dt
    from tidyfinance import compute_rolling_value
    rng = np.random.default_rng(42)
    df = pl.DataFrame({
        'date': pl.date_range(
            dt.date(2020, 1, 1), dt.date(2021, 12, 1), '1mo', eager=True
        ),
        'value': rng.standard_normal(24),
    })
    df = df.with_columns(
        rolling_sd=pl.Series(compute_rolling_value(
            df,
            f=lambda x: x['value'].std(ddof=1),
            period='month',
            periods=4,
            min_obs=2,
        ))
    )
    ```
    """
    if data_options is None:
        data_options = {"date": "date"}
    if min_obs is None:
        min_obs = periods

    date_col = data_options.get("date")

    if not isinstance(date_col, str):
        raise ValueError(
            "'date' in data_options must be a single non-missing string."
        )

    if date_col not in data.columns:
        raise ValueError(f"'data' must contain a '{date_col}' column.")

    date_dtype = data.schema[date_col]
    if not (date_dtype == pl.Date or isinstance(date_dtype, pl.Datetime)):
        raise ValueError(f"The '{date_col}' column must be of datetime dtype.")

    if not isinstance(period, str):
        raise ValueError("'period' must be a single string.")

    period_offset_map = {
        "month": ("1mo", "mo"),
        "quarter": ("1q", "q"),
        "year": ("1y", "y"),
    }
    if period not in period_offset_map:
        raise ValueError("'period' must be one of 'month', 'quarter', 'year'.")
    every, unit = period_offset_map[period]

    buckets = data.get_column(date_col).dt.truncate(every)
    if periods - 1 > 0:
        start_buckets = buckets.dt.offset_by(f"-{periods - 1}{unit}")
    else:
        start_buckets = buckets

    float_cols = [
        name for name, dtype in data.schema.items() if dtype.is_float()
    ]

    n = data.height
    result = np.full(n, np.nan)

    # The user callback receives the window in the active backend's
    # frame type, so pandas-style callbacks keep working under the
    # default backend.
    as_pandas = get_backend() == "pandas"

    for i in range(n):
        in_window = (buckets >= start_buckets[i]) & (buckets <= buckets[i])
        window_data = data.filter(in_window).drop_nulls()
        if float_cols:
            window_data = window_data.filter(
                ~pl.any_horizontal([pl.col(c).is_nan() for c in float_cols])
            )
        if window_data.height >= min_obs:
            value = f(window_data.to_pandas() if as_pandas else window_data)
            result[i] = np.nan if value is None else value

    return result
