import logging
import os
from datetime import datetime, timedelta

import polars as pl

from ..utils.constants import PathLike
from .base_extractor import Extractor

logger = logging.getLogger(__name__)


class FG5Extractor(Extractor):
    """
    Extractor for Micro-g FG5/FG5X set txt file.
    """

    delimiter: str = "\t"
    file_extension = ".set.txt"
    datetime_format: str = "%m/%d/%Y %H:%M:%S"

    def _get_header_index(self, path: PathLike) -> int:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                if line.startswith("Set"):
                    return idx
        raise ValueError("Header line not found in CG6 file")

    def _extract_impl(self, path: PathLike) -> pl.DataFrame:
        """
        Load FG5 data into a DataFrame.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            header: list[str] = self.get_header(path)
            df: pl.DataFrame = pl.read_csv(
                path,
                separator=self.delimiter,
                skip_rows=self._get_header_index(path),
                ignore_errors=True,
            )
            df.columns = header

            # format = YYYY-DOY HH:MM:SS
            df = df.with_columns(
                (
                    pl.col("Year").cast(pl.String)
                    + "-"
                    + pl.col("DOY").cast(pl.String).str.zfill(3)
                    + " "
                    + pl.col("Time").cast(pl.String)
                )
                .str.strptime(pl.Datetime, "%Y-%j %H:%M:%S", strict=False)
                .alias("timestamp")
            )

            df = df.drop(["Year", "DOY", "Time"])
            cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
            df = df.select(cols)

            if df is None or df.height == 0:
                logger.warning("Dataframe is empty in file: %s", path)

            return df
        except Exception as e:
            raise RuntimeError(f"Error reading file: {e}") from e

    def get_header(self, path: PathLike) -> list[str]:
        """
        Retrieves the header (channels) from the drift file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        start = self._get_header_index(path)
        header_line = lines[start].strip()

        return header_line.split(self.delimiter)

    def get_end(self, path: PathLike) -> datetime:
        """
        Get the last timestamp from the drift file.
        """
        df = (
            pl.scan_csv(
                path,
                separator=self.delimiter,
                skip_rows=self._get_header_index(path),
                ignore_errors=True,
            )
            .tail(1)
            .collect()
        )

        if df.height == 0:
            raise ValueError(f"Empty dataframe in file: {path}")

        last_row = df.row(0)

        year = int(last_row[1])
        doy = int(last_row[2])
        time_str = str(last_row[3])

        date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        datetime_str = f"{date:%Y-%m-%d} {time_str}"

        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
