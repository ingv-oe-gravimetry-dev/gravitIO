import logging

import polars as pl

logger = logging.getLogger(__name__)


class Correction:
    """
    Base class for data extractors.
    """

    _baro_corr_factor: float = -0.33
    _baro_col_name: str | None = None
    _grav_col_name: str | None = None

    @property
    def baro_corr_factor(self) -> float:
        """
        Get barometric correction factor
        """
        return self._baro_corr_factor

    @baro_corr_factor.setter
    def baro_corr_factor(self, value: float) -> None:
        """
        Set the barometric correction factor
        """
        self._baro_corr_factor = value

    @property
    def baro_col_name(self) -> str | None:
        """
        Get barometer column name
        """
        return self._baro_col_name

    @baro_col_name.setter
    def baro_col_name(self, value: str) -> None:
        """
        Set the barometer column name
        """
        self._baro_col_name = value

    @property
    def grav_col_name(self) -> str | None:
        """
        Get gravity column name
        """
        return self._grav_col_name

    @grav_col_name.setter
    def grav_col_name(self, value: str) -> None:
        """
        Set gravity column name
        """
        self._grav_col_name = value

    def _column_index(self, df: pl.DataFrame, col_substr: str) -> int:
        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        matches: list = [col for col in df.columns if col_substr.strip().lower() == col.lower()]

        if len(matches) == 0:
            raise ValueError(f"No column name exactly matches '{col_substr}'. Available columns: {list(df.columns)}")

        if len(matches) > 1:
            raise ValueError(f"Multiple columns exactly match '{col_substr}': {matches}. Please refine the substring.")

        col_name: str = matches[0]
        loc = df.columns.index(col_name)

        return loc

    def apply_baro_corr(
        self,
        df: pl.DataFrame,
        grav_col_index: int | None = None,
        baro_col_index: int | None = None,
        baro_correction_factor: float = _baro_corr_factor,
    ) -> pl.DataFrame:
        """
        Applies a barometric correction to the specified gravity and barometer columns.
        If both column numbers are None, uses default column names.
        """

        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        logger.debug("Applying barometric correction to gravity and barometer columns.")

        if grav_col_index is None and baro_col_index is None:
            # Require column names in this case
            if not hasattr(self, "_grav_col_name") or not hasattr(self, "_baro_col_name"):
                logger.error("Default column names not defined in class.")
                raise ValueError("Default column names not defined in class.")

            if self._grav_col_name is None or self._baro_col_name is None:
                logger.error("Column names for gravity or barometer not set.")
                raise ValueError("Column names for gravity or barometer not set.")

            try:
                grav_col_index = self._column_index(df, self._grav_col_name)
                baro_col_index = self._column_index(df, self._baro_col_name)
            except KeyError as e:
                logger.error("Column name not found in dataframe: %s", e)
                raise ValueError(f"Column name not found: {e}") from e

        elif grav_col_index is None or baro_col_index is None:
            logger.error("Either specify both column numbers or none.")
            raise ValueError("Either specify both column numbers or none.")

        if grav_col_index >= len(df.columns) or baro_col_index >= len(df.columns):
            logger.error(
                "Invalid column numbers: grav_col_index=%s, baro_col_index=%s.",
                grav_col_index,
                baro_col_index,
            )
            raise ValueError(
                f"Invalid column numbers: grav_col_index={grav_col_index}, baro_col_index={baro_col_index}."
            )

        grav_col_name = df.columns[grav_col_index]
        baro_col_name = df.columns[baro_col_index]

        df = df.with_columns(
            (pl.col(baro_col_name).cast(pl.Float64, strict=False) * baro_correction_factor).alias(baro_col_name)
        )

        df = df.with_columns(
            (pl.col(grav_col_name).cast(pl.Float64, strict=False) - pl.col(baro_col_name)).alias(grav_col_name)
        )

        logger.debug(
            "Barometric correction applied using gravity column %s and barometer column %s.",
            grav_col_index,
            baro_col_index,
        )

        return df
