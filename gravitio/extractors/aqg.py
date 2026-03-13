import logging
import os
from pathlib import Path

import polars as pl

from .csv import CSVExtractor

logger = logging.getLogger(__name__)
PathLike = str | Path


class AQGExtractor(CSVExtractor):
    """
    A class for extracting data from CSV AQG files.
    """

    datetime_format: str = "%Y/%m/%d %H:%M:%S"

    def _extract_impl(self, path: PathLike) -> pl.DataFrame:
        """
        Loads data from a CSV AQG file into a polars DataFrame.
        """
        if not os.path.exists(path):
            logger.error("File not found: '%s'.", path)
            raise FileNotFoundError(f"The file '{path}' does not exist.")

        header: list[str] = self.get_header(path)

        try:
            df: pl.DataFrame = pl.read_csv(
                path,
                separator=self.delimiter,
                has_header=False,
                skip_rows=1,
                ignore_errors=True,
            )

            if df is None or df.height == 0:
                logger.warning("Dataframe is empty in file: %s", path)
                return df

            df.columns = header

            logger.debug(
                "Data loaded successfully from '%s', total valid rows: %s.",
                path,
                df.height,
            )

            df = df.with_columns(
                (pl.col("date (utc)").cast(pl.String) + " " + pl.col("time (utc)").cast(pl.String))
                .str.strptime(pl.Datetime, self.datetime_format, strict=False)
                .alias("timestamp")
            )

            df = df.drop(["timestamp (s)", "date (utc)", "time (utc)"])

            # Reorder 'timestamp' to be the first column
            cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
            df = df.select(cols)

            return df

        except FileNotFoundError as e:
            logger.error("File not found: %s", e)
            raise

        except Exception as e:
            logger.error("An error occurred while loading data: %s", e)
            raise
