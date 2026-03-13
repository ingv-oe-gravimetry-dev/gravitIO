import logging
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)


def dropnan(
    df: pl.DataFrame,
    how: Literal["all", "any"] = "all",
    subset: list[str] | None = None,
) -> pl.DataFrame:
    """
    Drop rows with missing values from a DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame to clean.
        how (Literal["all", "any"]):
            Drop rows if 'any' or 'all' values are NaN.
        subset (list[str], optional):
            Columns to check for NaNs. Defaults to all columns.

    Returns:
        pl.DataFrame: Cleaned DataFrame.
    """

    if not isinstance(df, pl.DataFrame):
        logger.error("Input is not a polars DataFrame.")
        raise TypeError("Input must be a polars DataFrame.")

    if df is None or df.height == 0:
        logger.debug("DataFrame is empty. Nothing to drop.")
        return df

    if subset is None:
        return df

    initial_rows = df.height

    if how == "any":
        cleaned_df = df.drop_nulls(subset=subset)
    else:  # how == "all"
        cleaned_df = df.filter(~pl.all_horizontal(pl.col(subset).is_null()))

    dropped_rows = initial_rows - cleaned_df.height

    logger.debug(
        "Dropped %s rows from DataFrame (initial: %s, final: %s).",
        dropped_rows,
        initial_rows,
        cleaned_df.height,
    )

    return cleaned_df
