import logging

from .base_correction import Correction

logger = logging.getLogger(__name__)


class AQGCorrection(Correction):
    """
    A class for applying corrections to iGrav data
    """
