import logging
from pathlib import Path
from typing import TypedDict

import polars as pl

from .constants import PathLike

logger = logging.getLogger(__name__)


class ChannelConfigNotRequired(TypedDict, total=False):
    """channel configuration"""

    conversion: float  # opzionale


class ChannelConfig(ChannelConfigNotRequired, total=True):
    """channel configuration"""

    column: int
    channel: str


def get_sample_type(dtype: pl.DataType) -> str:
    """Map Polars data types to MiniSEED sample types."""
    if dtype in [pl.Int32]:
        return "i"
    elif dtype in [pl.Float32]:
        return "f"
    elif dtype in [pl.Float64]:
        return "d"
    elif dtype in [pl.Utf8, pl.String]:
        return "t"
    raise ValueError(f"Unsupported dtype for MiniSEED: {dtype}")


def calculate_sample_rate(df: pl.DataFrame, time_col: str) -> float:
    """Calculate sample rate in Hz based on the median time difference."""
    median_us = df.select(pl.col(time_col).diff().drop_nulls().median().dt.total_microseconds()).item()

    if median_us is None or median_us == 0:
        raise ValueError("Invalid or identical timestamps, cannot calculate Sample Rate.")

    median_us = float(median_us)

    return 1_000_000.0 / median_us


def add_time_and_segment_features(
    df: pl.DataFrame,
    time_col: str,
    sample_rate: float,
) -> pl.DataFrame:
    """Enrich dataframe with daily features and segment IDs for gaps."""
    tolerance_us = int((1.5 / sample_rate) * 1_000_000)

    return df.with_columns(
        [
            pl.col(time_col).dt.date().alias("_date"),
            pl.col(time_col).dt.year().alias("_year"),
            pl.col(time_col).dt.ordinal_day().alias("_jday"),
            (pl.col(time_col).diff().dt.total_microseconds() > tolerance_us)
            .fill_null(True)
            .cast(pl.Int32)
            .cum_sum()
            .alias("_segment_id"),
        ]
    )


def build_fdsn_source_id(network: str, station: str, location: str, channel: str) -> str:
    """Format the FDSN Source ID string required by MS3TraceList."""
    if len(channel) == 3:
        return f"FDSN:{network}_{station}_{location}_{channel[0]}_{channel[1]}_{channel[2]}"
    return f"FDSN:{network}_{station}_{location}_{channel}"


def setup_output_directory(out_dir: PathLike) -> Path:
    """Ensure the output directory exists and is strictly a directory."""
    base_dir = Path(out_dir)

    if base_dir.is_file():
        base_dir = base_dir.parent

    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir
