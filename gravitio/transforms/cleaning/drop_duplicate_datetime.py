import logging

import polars as pl

logger = logging.getLogger(__name__)


def drop_duplicate_datetime(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Drops rows with datetime values that are not strictly increasing from a DataFrame.

    Args:
        df (pl.DataFrame): DataFrame to clean.
        column (str): Column name containing datetime values.

    Returns:
        pl.DataFrame: Cleaned DataFrame.
    """

    if column not in df.columns:
        logger.error("DataFrame must contain '%s' column.", column)
        raise ValueError(f"DataFrame must contain '{column}' column.")

    if df is None or df.height == 0:
        logger.debug("DataFrame is empty. Returning as is.")
        return df

    # Ensure datetime type (assuming string or existing datetime)
    df = df.with_columns(pl.col(column).cast(pl.Datetime, strict=False))

    # Mark rows where datetime is strictly increasing (first row is True)
    is_valid = (pl.col(column) > pl.col(column).shift()).fill_null(True)

    cleaned_df = df.filter(is_valid)

    dropped_count = df.height - cleaned_df.height

    if dropped_count > 0:
        logger.debug("Dropped %s non-increasing datetime rows.", dropped_count)

    return cleaned_df
