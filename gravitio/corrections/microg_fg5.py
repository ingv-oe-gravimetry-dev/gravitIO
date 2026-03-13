import logging

import polars as pl

logger = logging.getLogger(__name__)


class FG5Correction:
    """
    A class for applying corrections to iGrav data
    """

    def add_self_attraction(
        self,
        df: pl.DataFrame,
        self_attraction_factor: float = 0.0,
    ) -> pl.DataFrame:
        """
        Adds the self-attraction correction to the data.
        """
        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        logger.debug("Adding self-attraction factor.")

        if "selfattraction" not in df.columns:
            df = df.with_columns(pl.lit(self_attraction_factor).alias("selfattraction"))
        else:
            df = df.with_columns(pl.col("selfattraction").fill_null(self_attraction_factor))
        return df

    def add_self_diffraction(
        self,
        df: pl.DataFrame,
        self_diffraction_factor: float = 0.0,
    ) -> pl.DataFrame:
        """
        Adds the self-diffraction correction to the data.
        """
        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        logger.debug("Adding self-diffraction factor.")

        if "selfdiffraction" not in df.columns:
            df = df.with_columns(pl.lit(self_diffraction_factor).alias("selfdiffraction"))
        else:
            df = df.with_columns(pl.col("selfdiffraction").fill_null(self_diffraction_factor))
        return df

    def apply_gravity_corr(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies the gravity correction to the data.
        """
        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        logger.debug("Applying gravity corrections")

        df = df.with_columns(
            (pl.col("gravity") + pl.col("selfattraction") + pl.col("selfdiffraction")).alias("gravitycorr")
        )
        return df

    def mean_set(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute set mean.
        """
        if df is None or df.height == 0:
            logger.error("Dataframe is empty.")
            raise ValueError("Dataframe is empty.")

        if "gravitycorr" in df.columns:
            gravity_corr_mean = df.get_column("gravitycorr").mean()
        else:
            gravity_corr_mean = float("nan")

        return pl.DataFrame(
            {
                "datetime_start": [df.get_column("timestamp")[0]],
                "datetime_end": [df.get_column("timestamp")[-1]],
                "gravity_mean": [df.get_column("gravity").mean()],
                "gravitycorr_mean": [gravity_corr_mean],
                "set_number": [df.get_column("gravity").drop_nulls().len()],
            }
        )
