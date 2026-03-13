import logging
from datetime import datetime

import polars as pl

logger = logging.getLogger(__name__)


class CGXCorrection:
    """
    A class for applying corrections to iGrav data
    """

    def __init__(self) -> None:
        self.grav_col_name: str = "corrgrav"

    def mean_by_station(self, df: pl.DataFrame | None) -> pl.DataFrame:
        """
        Compute daily mean CorrGrav for each station.
        """

        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        df = df.with_columns(pl.col("timestamp").dt.truncate("1d").alias("timestamp"))

        return (
            df.group_by(["station", "timestamp"])
            .agg(
                pl.col(self.grav_col_name).cast(pl.Float64, strict=False).mean().alias(f"{self.grav_col_name}_mean"),
                pl.len().alias("measures_number"),
            )
            .sort(["station", "timestamp"])
        )

    def mean_by_consecutive_measure(self, df: pl.DataFrame | None) -> pl.DataFrame:
        """
        Compute mean CorrGrav for consecutive measurements
        of the same station.

        A new block starts when the station changes.
        """

        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        df = df.sort("timestamp")

        # Create a block ID that increments when the station changes
        df = df.with_columns(
            (pl.col("station") != pl.col("station").shift()).fill_null(True).cum_sum().alias("block_id")
        )

        return (
            df.group_by("block_id")
            .agg(
                pl.col("station").first().alias("station"),
                pl.col("timestamp").first().alias("datetime_start"),
                pl.col("timestamp").last().alias("datetime_end"),
                pl.len().alias("measures_numer"),
                pl.col(self.grav_col_name).cast(pl.Float64, strict=False).mean().alias("corrgrav_mean"),
            )
            .sort("block_id")
            .drop("block_id")
        )

    def station_remap(self, df: pl.DataFrame | None, mapping: list[dict[str, str]]) -> pl.DataFrame:
        """
        Remap station names based on a list of intervals.

        Each element in `mapping` should be a dict:
        {
            "station": new_station_name,
            "from": "YYYY-MM-DD HH:MM:SS",
            "to":   "YYYY-MM-DD HH:MM:SS"
        }

        Only consecutive measurements falling within each interval will be renamed.
        """

        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        station_expr = pl.col("station")

        for entry in mapping:
            new_station = entry["station"]
            start = datetime.fromisoformat(entry["from"].replace(" ", "T"))
            end = datetime.fromisoformat(entry["to"].replace(" ", "T"))

            # Apply the new station name only to rows within the interval
            mask = (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
            station_expr = pl.when(mask).then(pl.lit(new_station)).otherwise(station_expr)

        df = df.with_columns(station_expr.alias("station"))
        return df

    def remove_outlier(self, df: pl.DataFrame | None, sigma: float = 3.0) -> pl.DataFrame:
        """
        Remove outliers using a 3-sigma criterion applied to
        consecutive measurements of the same station.
        """

        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        df = df.sort("timestamp")

        # Create block ID for consecutive measurements of same station
        df = df.with_columns(
            (pl.col("station") != pl.col("station").shift()).fill_null(True).cum_sum().alias("block_id")
        )

        grav_expr = pl.col(self.grav_col_name).cast(pl.Float64, strict=False)
        mean_expr = grav_expr.mean().over("block_id")
        std_expr = grav_expr.std().over("block_id")

        valid_mask = (grav_expr - mean_expr).abs() <= sigma * std_expr
        fallback_mask = std_expr.fill_null(0.0) == 0.0

        return df.filter(valid_mask | fallback_mask).drop("block_id")
