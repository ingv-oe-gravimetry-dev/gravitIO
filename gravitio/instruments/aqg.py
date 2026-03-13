from ..corrections.aqg import AQGCorrection
from ..exporters.base_exporter import Exporter
from ..extractors.aqg import AQGExtractor


class AQG(AQGExtractor, AQGCorrection, Exporter):
    """
    Combined class for AQG data extraction, corrections and export.
    """


__all__ = ["AQG"]
