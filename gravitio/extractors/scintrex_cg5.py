import datetime
import logging
import re
from pathlib import Path

import polars as pl

from ..utils.constants import PathLike
from .base_extractor import Extractor

logger = logging.getLogger(__name__)


class CG5Extractor(Extractor):
    """
    Fast extractor for Scintrex CG5 files (no python engine).
    """

    delimiter: str = r"\s+"
    file_extension: str = ".txt"
    datetime_format: str = "%Y/%m/%d %H:%M:%S"

    def get_header(self, path: PathLike) -> list[str]:
        path = Path(path)

        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("/-"):
                    header_line = line.strip().lstrip("/")
                    break
            else:
                raise ValueError("Header line not found in CG5 file")

        header_line = re.sub(r"-+", " ", header_line)

        header = [
            col.rstrip(".")  # remove the trailing dot
            for col in header_line.strip().lower().split()
        ]

        return header

    def _extract_impl(self, path: PathLike) -> pl.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            header_idx = None
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip().startswith("/-"):
                        header_idx = i
                        break

            if header_idx is None:
                raise ValueError("Header line not found")

            header = self.get_header(path)

            df = pl.read_csv(
                path,
                has_header=False,
                skip_rows=header_idx + 1,
                separator="\x1f",
                new_columns=["raw"],
                truncate_ragged_lines=True,
                ignore_errors=True,
            )

            df_cleaned = df.with_columns(pl.col("raw").str.strip_chars().str.replace_all(self.delimiter, ",")).filter(
                pl.col("raw") != ""
            )

            df_split = df_cleaned.select(pl.col("raw").str.split_exact(",", len(header) - 1).alias("fields")).unnest(
                "fields"
            )

            rename_map = {f"field_{i}": col for i, col in enumerate(header)}
            df_final = df_split.rename(rename_map)

            df_final = df_final.with_columns(
                (pl.col("date").cast(pl.String) + " " + pl.col("time").cast(pl.String))
                .str.strptime(pl.Datetime, self.datetime_format, strict=False)
                .alias("timestamp")
            )

            df_final = df_final.drop(["date", "time", "dec.time+date"])
            cols = ["timestamp"] + [c for c in df_final.columns if c != "timestamp"]
            df_final = df_final.select(cols)

            numeric_cols = [col for col in header if col not in ("date", "time", "dec.time+date", "timestamp")]
            df_final = df_final.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])

            return df_final

        except Exception as e:
            logger.error("Error while reading CG5 file: %s", e)
            raise

    def get_end(self, path: PathLike) -> datetime.datetime:
        path = Path(path)

        header_idx = None
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip().startswith("/------LINE"):
                    header_idx = i
                    break

        if header_idx is None:
            raise ValueError("Header line not found")

        df = pl.read_csv(
            path,
            has_header=False,
            skip_rows=header_idx + 1,
            separator="\x1f",
            new_columns=["raw"],
            truncate_ragged_lines=True,
            ignore_errors=True,
        )

        valid_lines = df.get_column("raw").str.strip_chars().filter(df.get_column("raw").str.strip_chars() != "")

        if valid_lines.len() == 0:
            raise ValueError("No valid data found in file")

        last_line = str(valid_lines[-1])
        last_line_clean = re.sub(self.delimiter, ",", last_line.strip())
        parts = last_line_clean.split(",")

        time_str = parts[-2]
        date_str = parts[-1]

        datetime_str = f"{date_str} {time_str}"
        return datetime.datetime.strptime(datetime_str, self.datetime_format)
