import gzip
import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
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

    file_extension: str | Sequence[str]

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

    def _file_extensions(self) -> tuple[str, ...]:
        """Return file extensions as a tuple regardless of class-level declaration."""
        if isinstance(self.file_extension, str):
            return (self.file_extension,)
        return tuple(self.file_extension)

    def _get_files_from_gz(self, gz_path: PathLike, temp_dir: Path | None = None) -> Iterator[str]:
        """
        Generalizes reading a compressed .gz file for any file type.
        This function looks for lines that match the file extension.
        """

        gz_path = Path(gz_path)

        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir()) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        filename: str = Path(gz_path).stem
        out_path: Path = temp_dir / filename

        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        extensions = self._file_extensions()
        if out_path.name.endswith(extensions):
            yield str(out_path)

    def _get_files_from_tar_gz(self, tar_gz_path: PathLike, temp_dir: Path | None = None) -> Iterator[str]:
        """
        Generalizes extracting files from a .tar.gz archive based on the file extension.
        """

        tar_gz_path = Path(tar_gz_path)

        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir()) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_resolved = temp_dir.resolve()
        extensions = self._file_extensions()

        with tarfile.open(tar_gz_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(extensions):
                    continue

                output_path = (temp_dir / member.name).resolve()
                if os.path.commonpath([str(temp_dir_resolved), str(output_path)]) != str(temp_dir_resolved):
                    logger.warning("Skipping unsafe archive member path: %s", member.name)
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue

                with source, open(output_path, "wb") as destination:
                    shutil.copyfileobj(source, destination)
                yield str(output_path)

    def _get_files_from_zip(self, zip_path: PathLike, temp_dir: Path | None = None) -> Iterator[str]:
        """
        Generalizes extracting files from a .zip archive based on the file extension.
        """

        zip_path = Path(zip_path)

        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir()) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_resolved = temp_dir.resolve()
        extensions = self._file_extensions()

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.is_dir():
                    continue

                member_name = zip_info.filename.replace("\\", "/")
                if not member_name.endswith(extensions):
                    continue

                output_path = (temp_dir / Path(member_name)).resolve()
                if os.path.commonpath([str(temp_dir_resolved), str(output_path)]) != str(temp_dir_resolved):
                    logger.warning("Skipping unsafe archive member path: %s", zip_info.filename)
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(zip_info, "r") as source, open(output_path, "wb") as destination:
                    shutil.copyfileobj(source, destination)
                yield str(output_path)

    def file_list(self, path: PathLike, temp_dir: Path | None = None) -> list[Path]:
        """
        Returns a list of files based on the provided path and file extensions,
        including files inside .gz, .tar.gz, and .zip archives.
        """

        logger.debug("Scanning for files in: %s with extensions %s", path, self.file_extension)

        path = Path(path)
        if temp_dir is not None:
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        extensions = self._file_extensions()

        if path.is_file():
            if path.name.endswith(extensions):
                return [path]

            elif path.suffix == ".gz" and not path.name.endswith(".tar.gz"):
                return list(map(Path, self._get_files_from_gz(path, temp_dir=temp_dir)))

            elif path.name.endswith(".tar.gz"):
                return list(map(Path, self._get_files_from_tar_gz(path, temp_dir=temp_dir)))

            elif path.suffix == ".zip":
                return list(map(Path, self._get_files_from_zip(path, temp_dir=temp_dir)))

            logger.error("Invalid file format: %s", path)
            return []

        if path.is_dir():
            for file in path.rglob("*"):
                if file.name.endswith(extensions):
                    files.append(file)
                elif file.suffix == ".gz" and not file.name.endswith(".tar.gz"):
                    files.extend(map(Path, self._get_files_from_gz(file, temp_dir=temp_dir)))
                elif file.name.endswith(".tar.gz"):
                    files.extend(map(Path, self._get_files_from_tar_gz(file, temp_dir=temp_dir)))
                elif file.suffix == ".zip":
                    files.extend(map(Path, self._get_files_from_zip(file, temp_dir=temp_dir)))

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
        Convenience API: load all files and return a single merged DataFrame.

        This method reads every matching file, concatenates all per-file DataFrames
        in memory, sorts by ``timestamp``, and optionally applies cleaning.
        It is simple to use, but can require significant RAM on very large datasets.

        For file-by-file processing (progress bars, incremental export, lower memory
        peaks), use ``extract_iter`` instead.
        """

        if path is None:
            raise ValueError("path must be provided.")

        path = Path(path)

        with tempfile.TemporaryDirectory(prefix="gravitio_") as tmp_dir:
            files_to_process = self.file_list(path, temp_dir=Path(tmp_dir))
            total_size_mb = sum(os.path.getsize(f) for f in files_to_process) / (1024 * 1024)

            if total_size_mb > 500:
                logger.warning(
                    "LARGE DATASET DETECTED: %s files totaling %.2f MB.",
                    len(files_to_process),
                    total_size_mb,
                )

            # Eager mode: collect all per-file DataFrames before concatenation.
            frames = [self._extract_impl(file) for file in files_to_process]
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
        Streaming API: yield one ``(file, DataFrame)`` pair at a time.

        Unlike ``extract``, this method does not concatenate all frames into one
        final DataFrame. It is intended for incremental workflows (GUI progress,
        per-file error handling, chunked processing, lower peak memory usage).
        """

        path = Path(path)
        with tempfile.TemporaryDirectory(prefix="gravitio_") as tmp_dir:
            files_to_process = self.file_list(path, temp_dir=Path(tmp_dir))

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
