import datetime
import logging
import re
from pathlib import Path

import polars as pl

from ..utils.constants import PathLike
from .base_extractor import Extractor

logger = logging.getLogger(__name__)


class IGravExtractor(Extractor):
    """
    Extractor for iGrav TSF files.
    """

    delimiter: str = r"\s{2,}"
    datetime_format: str = "%Y %m %d %H %M %S"
    file_extension: str = ".tsf"

    def get_header(self, path: PathLike) -> list[str]:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        start_idx, end_idx = self.get_section_start_end_indexes(path, "[CHANNELS]")
        if start_idx == -1 or end_idx == -1:
            return []

        channels = [line.strip().lower().split(":")[-1] for line in lines[start_idx:end_idx]]
        return ["timestamp"] + [c for c in channels if c.lower() != "timestamp"]

    def get_section_start_index(self, path: PathLike, section_tag: str) -> int:
        """
        Return the index of the first line containing the specified section tag in the file at the given path.
        If the section tag is not found, return -1.
        """
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if section_tag in line:
                    return i + 1
        return -1

    def get_section_start_end_indexes(self, path: PathLike, section_tag: str) -> tuple[int, int]:
        """
        Return the start and end indices of the section containing the specified section tag in the file
        at the given path. If the section tag is not found, return (-1, -1).
        The start index is the index of the first line containing the section tag,
        the end index is the index of the first empty line after the section tag.
        """

        start_idx = self.get_section_start_index(path, section_tag)
        if start_idx == -1:
            return -1, -1

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        end_idx = start_idx
        while end_idx < len(lines) and lines[end_idx].strip():
            end_idx += 1
        return start_idx, end_idx

    def _extract_impl(self, path: PathLike) -> pl.DataFrame:
        """
        This extractor is specifically designed to safely parse these tricky sections
        and return a clean, tabular Polars DataFrame.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            data_start_idx = self.get_section_start_index(path, "[DATA]")
            if data_start_idx == -1:
                raise ValueError(f"No [DATA] section found in file: {path}")

            header = self.get_header(path)

            df = pl.read_csv(
                path,
                has_header=False,
                skip_rows=data_start_idx,
                separator="\x1f",
                new_columns=["raw"],
                truncate_ragged_lines=True,
                ignore_errors=True,
            )

            df_cleaned = df.with_columns(
                pl.col("raw")
                .str.replace_all("\x00", "")
                .str.strip_chars()
                .str.replace_all(r"\s{2,}", ",")
                .str.strip_chars_end(",")
            ).filter(pl.col("raw") != "")

            df_split = df_cleaned.select(pl.col("raw").str.split_exact(",", len(header) - 1).alias("fields")).unnest(
                "fields"
            )

            rename_map = {f"field_{i}": col for i, col in enumerate(header)}
            df_final = df_split.rename(rename_map)

            df_final = df_final.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, self.datetime_format, strict=False)
            )

            numeric_cols = [col for col in header if col != "timestamp"]
            df_final = df_final.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])

            return df_final

        except FileNotFoundError as e:
            logger.error("File not found: %s", e)
            raise

        except Exception as e:
            logger.error("An error occurred while loading data: %s", e)
            raise

    def get_end(self, path: PathLike) -> datetime.datetime:
        """
        Restituisce l'ultimo timestamp valido nel file TSF.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            data_start_idx = self.get_section_start_index(path, "[DATA]")
            if data_start_idx == -1:
                raise ValueError(f"No [DATA] section found in file: {path}")

            df = pl.read_csv(
                path,
                has_header=False,
                skip_rows=data_start_idx,
                separator="\x1f",
                new_columns=["raw"],
                truncate_ragged_lines=True,
                ignore_errors=True,
            )

            raw_col = df.get_column("raw")
            valid_lines = raw_col.str.strip_chars().filter(raw_col.str.strip_chars() != "")

            if valid_lines.len() == 0:
                raise ValueError("No valid data found in file")

            last_line = valid_lines[-1]
            last_line_clean = str(last_line).replace("\x00", "").strip()
            last_line_clean = re.sub(r"\s{2,}", ",", last_line_clean)
            ts_str = last_line_clean.split(",")[0]

            return datetime.datetime.strptime(ts_str, self.datetime_format)

        except FileNotFoundError as e:
            logger.error("File not found: %s", e)
            raise

        except Exception as e:
            logger.error("An error occurred while loading data: %s", e)
            raise
