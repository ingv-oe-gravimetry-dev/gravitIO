import logging
import os
from datetime import datetime
from pathlib import Path

import polars as pl

from ..utils.constants import PathLike
from .base_extractor import Extractor

logger = logging.getLogger(__name__)


class CSVExtractor(Extractor):
    """
    Extractor for CSV files.
    """

    delimiter = ","
    file_extension = ".csv"
    datetime_format = "%Y-%m-%d %H:%M:%S"

    def get_header(self, path: PathLike) -> list[str]:
        """
        Get CSV header (column names).
        """
        path = Path(path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, encoding="utf-8") as f:
            header_line: str = f.readline().strip()
            return header_line.lower().split(self.delimiter)

    def _extract_impl(self, path: PathLike) -> pl.DataFrame:
        """
        Load CSV data into a DataFrame.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            return pl.read_csv(path, separator=self.delimiter, ignore_errors=True)
        except Exception as e:
            raise RuntimeError(f"Error reading file: {e}") from e

    def get_end(self, path: PathLike) -> datetime:
        """
        Get the last timestamp from the CSV file using Polars LazyFrame.
        """
        df = pl.scan_csv(path, separator=self.delimiter, ignore_errors=True).tail(1).collect()

        if df.height == 0:
            raise ValueError(f"Empty dataframe in file: {path}")

        last_row = df.row(0)
        last_date = last_row[1]
        last_time = last_row[2]

        datetime_str = f"{last_date} {last_time}"
        return datetime.strptime(datetime_str, self.datetime_format)
