import gzip
import logging
import os
import shutil
import tarfile
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import polars as pl

from ..transforms.cleaning.drop_duplicate_datetime import drop_duplicate_datetime
from ..transforms.cleaning.drop_nan import dropnan
from ..utils.constants import PathLike

logger = logging.getLogger(__name__)


class Extractor(ABC):
    """
    Base class for data extractors.
    """

    file_extension: str

    @abstractmethod
    def _extract_impl(self, path: PathLike) -> pl.DataFrame:
        """
        A base class for extractors that read data into a Polars DataFrame.
        """

        raise NotImplementedError

    @abstractmethod
    def get_end(self, path: PathLike) -> datetime:
        """
        Get the end date from the file.
        """

        raise NotImplementedError

    @abstractmethod
    def get_header(self, path: PathLike) -> list[str]:
        """
        Get the header from the file.
        """

        raise NotImplementedError

    def _get_files_from_gz(self, gz_path: PathLike) -> Iterator[str]:
        """
        Generalizes reading a compressed .gz file for any file type.
        This function looks for lines that match the file extension.
        """

        gz_path = Path(gz_path)

        temp_dir: Path = Path(tempfile.gettempdir()) / "temp"
        temp_dir.mkdir(exist_ok=True)

        filename: str = Path(gz_path).stem
        out_path: Path = temp_dir / filename

        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        if any(out_path.name.endswith(ext) for ext in self.file_extension):
            yield str(out_path)

    def _get_files_from_tar_gz(self, tar_gz_path: PathLike) -> Iterator[str]:
        """
        Generalizes extracting files from a .tar.gz archive based on the file extension.
        """

        tar_gz_path = Path(tar_gz_path)

        temp_dir: Path = Path(tempfile.gettempdir()) / "temp"
        temp_dir.mkdir(exist_ok=True)

        with tarfile.open(tar_gz_path, "r:gz") as tar:
            for member in tar.getmembers():
                if any(member.name.endswith(ext) for ext in self.file_extension):
                    tar.extract(member, path=temp_dir)
                    yield str(temp_dir / member.name)

    def file_list(self, path: PathLike) -> list[Path]:
        """
        Returns a list of files based on the provided path and file extensions,
        including files inside .gz and .tar.gz archives.
        """

        logger.debug("Scanning for files in: %s with extensions %s", path, self.file_extension)

        path = Path(path)
        files: list[Path] = []

        if path.is_file():
            if path.name.endswith(self.file_extension):
                return [path]

            elif path.suffix == ".gz" and not path.name.endswith(".tar.gz"):
                return list(map(Path, self._get_files_from_gz(path)))

            elif path.name.endswith(".tar.gz"):
                return list(map(Path, self._get_files_from_tar_gz(path)))

            logger.error("Invalid file format: %s", path)
            return []

        if path.is_dir():
            for file in path.rglob("*"):
                if file.name.endswith(self.file_extension):
                    files.append(file)
                elif file.suffix == ".gz" and not file.name.endswith(".tar.gz"):
                    files.extend(map(Path, self._get_files_from_gz(file)))
                elif file.name.endswith(".tar.gz"):
                    files.extend(map(Path, self._get_files_from_tar_gz(file)))

            return files

        logger.error("Invalid path: %s", path)
        return []

    def extract(
        self,
        path: PathLike | None = None,
        drop_duplicate_dt: bool = True,
        drop_nan: bool = True,
        nan_subset: list[str] | None = None,
    ) -> pl.DataFrame | None:
        """
        Extract data from files and optionally clean the resulting DataFrame.
        """

        if path is None:
            raise ValueError("path must be provided.")

        path = Path(path)

        files_to_process = self.file_list(path)
        total_size_mb = sum(os.path.getsize(f) for f in files_to_process) / (1024 * 1024)

        if total_size_mb > 500:
            logger.warning(
                "LARGE DATASET DETECTED: %s files totaling %.2f MB.",
                len(files_to_process),
                total_size_mb,
            )

        frames = [df for _, df in self.extract_iter(path)]
        if not frames:
            return None

        df = pl.concat(frames, how="vertical_relaxed")
        df = df.sort("timestamp")

        if drop_duplicate_dt:
            df = drop_duplicate_datetime(df, column="timestamp")

        if drop_nan:
            df = dropnan(df, subset=nan_subset)

        return df

    def extract_iter(
        self,
        path: PathLike,
        drop_duplicate_dt: bool = False,
        drop_nan: bool = False,
        nan_subset: list[str] | None = None,
    ) -> Iterator[tuple[PathLike, pl.DataFrame]]:
        """
        Extract data from file(s) one by one.
        """

        path = Path(path)
        files_to_process = self.file_list(path)

        if not files_to_process:
            logger.error("No data found in files at %s", path)
            return

        for file in files_to_process:
            df: pl.DataFrame = self._extract_impl(file)

            if drop_duplicate_dt:
                df = drop_duplicate_datetime(df, column="timestamp")

            if drop_nan:
                df = dropnan(df, subset=nan_subset)

            yield file, df
