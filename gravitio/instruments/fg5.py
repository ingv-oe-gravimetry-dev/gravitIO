from ..corrections.microg_fg5 import FG5Correction
from ..exporters.base_exporter import Exporter
from ..extractors.microg_fg5 import FG5Extractor


class FG5(FG5Extractor, FG5Correction, Exporter):
    """
    Combined class for Micro-g FG5 / FG5X data extraction, corrections and export.
    """


__all__ = ["FG5"]
