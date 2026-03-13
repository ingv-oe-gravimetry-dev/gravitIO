import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


def exporter(extension: str) -> Callable:
    """
    A decorator for data exporters.

    Wraps a function with input validation, path normalization, and ensures
    the output directory exists. Works with arbitrary positional and keyword
    arguments.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            if len(args) == 0:
                raise ValueError("Expected at least 'self' as first argument")

            self_obj = args[0]

            # ---------- EXTRACT df and path ----------
            df = kwargs.get("df", None)
            if len(args) > 1:
                df = args[1]

            if df is None:
                logger.error("No data (df) provided to export.")
                raise ValueError("No data (df) provided to export.")

            path = kwargs.get("path", None)
            if len(args) > 2:
                path = args[2]

            # ---------- PATH VALIDATION ----------
            if path is None:
                class_name: str = self_obj.__class__.__name__.lower()
                path = Path.cwd() / f"{class_name}_results{extension}"
            else:
                if not isinstance(path, (str, Path)):
                    raise TypeError(f"path must be str or Path, got {type(path).__name__}")
                path = Path(path)

            # ---------- EXTENSION NORMALIZATION ----------
            if path.suffix.lower() != extension.lower():
                path = path.with_suffix(extension)

            # ---------- DIRECTORY CREATION ----------
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.exception("Cannot create directory for %s", path)
                raise RuntimeError(f"Cannot create directory for {path}: {e}") from e

            # ---------- OPTIONAL TIMESTAMP FORMAT ----------
            datetime_format = kwargs.get("datetime_format", "%Y-%m-%d %H:%M:%S")
            if "timestamp" in df.columns:
                df = df.with_columns(pl.col("timestamp").dt.strftime(datetime_format).alias("timestamp"))

            # ---------- CALL ORIGINAL FUNCTION ----------
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.exception("Failed to export data to %s", path)
                raise RuntimeError(f"Failed to export data to {path}: {e}") from e

        return wrapper

    return decorator
