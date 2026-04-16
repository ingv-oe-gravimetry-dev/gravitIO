import logging

import polars as pl

from .base_correction import Correction

logger = logging.getLogger(__name__)


class IGravCorrection(Correction):
    """
    A class for applying corrections to iGrav data
    """

    def __init__(self) -> None:
        self.grav_col_name: str | None = "grav"
        self.baro_col_name: str | None = "baro-press"

    def apply_grav_conversion(
        self,
        df: pl.DataFrame,
        grav_col_index: int | None = None,
        conversion_factor: float = 1.0,
    ) -> pl.DataFrame:
        """
        Applies a gravity conversion factor to a specified column.
        """

        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        logger.debug("Applying conversion factor of %s to gravity column.", conversion_factor)

        if grav_col_index is None:
            grav_col_index = self._column_index(df, "Grav")

        if grav_col_index >= len(df.columns):
            logger.error("Invalid column number: grav_col_index=%s.", grav_col_index)
            raise ValueError(f"Invalid column number: grav_col_index={grav_col_index}.")

        grav_col_name = df.columns[grav_col_index]
        df = df.with_columns(
            (pl.col(grav_col_name).cast(pl.Float64, strict=False) * conversion_factor).alias(grav_col_name)
        )

        logger.debug("Gravity conversion applied successfully.")
        return df

    def apply_base_corrections(
        self,
        df: pl.DataFrame,
        grav_col_index: int | None = None,
        grav_conversion_factor: float = 1.0,
        baro_col_index: int | None = None,
        baro_corr_factor: float | None = None,
    ) -> pl.DataFrame:
        """
        Applies the base corrections to iGrav data.
        Applies the gravity conversion factor to the specified column
        and then applies the barometric correction.
        """

        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        logger.debug("Applying iGrav base corrections.")

        df = self.apply_grav_conversion(
            df,
            grav_col_index=grav_col_index,
            conversion_factor=grav_conversion_factor,
        )

        df = self.apply_baro_corr(
            df,
            grav_col_index=grav_col_index,
            baro_col_index=baro_col_index,
            baro_correction_factor=self._baro_corr_factor if baro_corr_factor is None else baro_corr_factor,
        )

        return df
