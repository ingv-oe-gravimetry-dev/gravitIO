import logging
import os
import re
from datetime import datetime

import polars as pl

from ..utils.constants import PathLike
from .base_extractor import Extractor

logger = logging.getLogger(__name__)


class CG6Extractor(Extractor):
    """
    Extractor for Scintrex CG6 files.
    """

    delimiter: str = "\t"
    file_extension = ".dat"
    datetime_format: str = "%m/%d/%Y %H:%M:%S"

    def get_header(self, path: PathLike) -> list[str]:
        """
        Retrieves the header (channels) from the drift file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        start = self._get_start(path)
        header_line = lines[start].strip()

        return header_line.lstrip("/").lower().split(self.delimiter)

    def _get_start(self, path: PathLike) -> int:
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line.startswith("/") and re.match(r"\/\w+", line):
                    return idx
        raise ValueError("Header line not found in CG6 file")

    def _extract_impl(self, path: PathLike) -> pl.DataFrame:
        """
        Load CG-6 data into a DataFrame.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            header: list[str] = self.get_header(path)
            df: pl.DataFrame = pl.read_csv(
                path,
                separator=self.delimiter,
                skip_rows=self._get_start(path),
                ignore_errors=True,
            )
            df.columns = header

            df = df.with_columns(
                (pl.col("date").cast(pl.String) + " " + pl.col("time").cast(pl.String))
                .str.strptime(pl.Datetime, self.datetime_format, strict=False)
                .alias("timestamp")
            )
            df = df.drop(["date", "time"])
            cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
            df = df.select(cols)

            if df is None or df.height == 0:
                logger.warning("Dataframe is empty in file: %s", path)

            return df
        except Exception as e:
            raise RuntimeError(f"Error reading file: {e}") from e

    def get_end(self, path: PathLike) -> datetime:
        """
        Get the last timestamp from the drift file.
        """
        df = (
            pl.scan_csv(path, separator=self.delimiter, skip_rows=self._get_start(path), ignore_errors=True)
            .tail(1)
            .collect()
        )

        if df.height == 0:
            raise ValueError(f"Empty dataframe in file: {path}")

        last_row = df.row(0)
        last_date = str(last_row[1])
        last_time = str(last_row[2])

        datetime_str = f"{last_date} {last_time}"
        return datetime.strptime(datetime_str, self.datetime_format)
