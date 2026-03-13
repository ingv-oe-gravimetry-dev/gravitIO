import logging

import polars as pl

logger = logging.getLogger(__name__)


def resample(df: pl.DataFrame, freq: int) -> pl.DataFrame:
    """
    Resample a DataFrame to a given frequency.

    Args:
        df (pl.DataFrame): The DataFrame to resample.
        freq (int): The frequency to resample to, in seconds.

    Returns:
        pl.DataFrame: The resampled DataFrame.
    """

    if not isinstance(df, pl.DataFrame):
        logger.error("Input is not a polars DataFrame.")
        raise TypeError("Input must be a polars DataFrame.")

    if df is None or df.height == 0:
        logger.error("Dataframe is empty.")
        raise ValueError("Dataframe is empty.")

    time_col = df.columns[0]
    df = df.sort(time_col)

    numeric_cols = []
    other_cols = []
    for col in df.columns:
        if col == time_col:
            continue
        if df.schema[col].is_numeric():
            numeric_cols.append(pl.col(col).mean())
        else:
            other_cols.append(pl.col(col).last())

    data_cols = [c for c in df.columns if c != time_col]

    resampled_df = df.group_by_dynamic(time_col, every=f"{freq}s").agg(numeric_cols + other_cols)

    if data_cols:
        resampled_df = resampled_df.filter(~pl.all_horizontal(pl.col(data_cols).is_null()))

    return resampled_df.select(df.columns)
