import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from pymseed import MS3TraceList, timestr2nstime

from ..utils.decorators import exporter
from ..utils.miniseed import (
    ChannelConfig,
    add_time_and_segment_features,
    build_fdsn_source_id,
    calculate_sample_rate,
    get_sample_type,
    setup_output_directory,
)

logger = logging.getLogger(__name__)
PathLike = str | Path

if TYPE_CHECKING:
    import pandas as pd


class Exporter:
    """
    Base class for data export
    """

    @classmethod
    def supported_formats(cls) -> list[str]:
        """Return sorted export format names based on available ``to_*`` methods."""
        formats: set[str] = set()

        for attr_name in dir(cls):
            if not attr_name.startswith("to_") or attr_name == "to_pandas":
                continue

            method = getattr(cls, attr_name)
            if not callable(method):
                continue

            format_name = attr_name.removeprefix("to_")
            if format_name == "excel":
                format_name = "xlsx"

            formats.add(format_name)

        return sorted(formats)

    @exporter(".parquet")
    def to_parquet(self, df: pl.DataFrame | None, path: PathLike) -> None:
        """Export data to Parquet file."""
        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")
        df.write_parquet(path)

    @exporter(".feather")
    def to_feather(self, df: pl.DataFrame | None, path: PathLike) -> None:
        """Export data to Feather file."""
        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")
        df.write_ipc(path)

    @exporter(".xlsx")
    def to_excel(self, df: pl.DataFrame | None, path: PathLike) -> None:
        """Export data to Excel file."""
        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")
        df.write_excel(path)

    @exporter(".csv")
    def to_csv(self, df: pl.DataFrame | None, path: PathLike) -> None:
        """Export data to CSV file."""
        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")
        df.write_csv(path, null_value="NaN")

    def to_pandas(self, df: pl.DataFrame | None) -> "pd.DataFrame":
        """Export data to Pandas DataFrame."""
        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        return df.to_pandas()

    def to_miniseed(
        self,
        df: pl.DataFrame | None,
        path: PathLike,
        network: str,
        station: str,
        channels: list[ChannelConfig],
        location: str = "",
        mseed_version: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        Export selected DataFrame columns to daily MiniSEED files.
        Orchestrates data cleaning, sample rate calculation, and file writing.
        """
        if mseed_version not in (2, 3):
            raise ValueError("mseed_version must be 2 or 3")

        time_column_index = kwargs.get("time_column_index", 0)

        if df is None or df.height < 2:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty or has insufficient data.")

        time_col_name = df.columns[time_column_index]
        df_clean = df.drop_nulls(subset=[time_col_name])

        # Calculate Sample Rate
        sample_rate = calculate_sample_rate(df_clean, time_col_name)

        # Handle gaps and dates
        df_processed = add_time_and_segment_features(df_clean, time_col_name, sample_rate)

        # Prepare output folder
        base_dir = setup_output_directory(path)

        for day_df in df_processed.partition_by("_date"):
            year = day_df["_year"][0]
            jday = day_df["_jday"][0]

            traces = MS3TraceList()
            has_data = False
            channel_names = []

            for ch in channels:
                col_idx = ch["column"]
                channel = ch["channel"]
                conversion = ch.get("conversion", None)

                col_name = df.columns[col_idx]
                channel_names.append(channel)

                source_id = build_fdsn_source_id(network, station, location, channel)

                original_dtype = df.schema[col_name]

                for segment_df in day_df.partition_by("_segment_id"):
                    segment_data = segment_df.drop_nulls(subset=[col_name])
                    if segment_data.height == 0:
                        continue

                    series = segment_data[col_name]

                    if conversion is not None:
                        series = series.cast(pl.Float64) * conversion
                        sample_type = "d"

                    elif original_dtype == pl.Int64 and conversion is None:
                        series = series.cast(pl.Float64)
                        sample_type = "d"

                    else:
                        sample_type = get_sample_type(original_dtype)

                    data = series.to_numpy().tolist()
                    block_start = segment_data[time_col_name][0]
                    start_ns = block_start.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"

                    traces.add_data(
                        sourceid=source_id,
                        data_samples=data,
                        sample_type=sample_type,
                        sample_rate=sample_rate,
                        start_time=timestr2nstime(start_ns),
                    )

                    has_data = True

            # Save daily SDS file
            if has_data:
                if len(channel_names) == 1:
                    filename = f"{network}.{station}.{location}.{channel_names[0]}.D.{year}.{jday:03d}.mseed"
                else:
                    filename = f"{network}.{station}.{location}.D.{year}.{jday:03d}.mseed"
                daily_path = base_dir / filename
                traces.to_file(str(daily_path), format_version=mseed_version)
