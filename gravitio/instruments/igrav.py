from ..corrections.igrav import IGravCorrection
from ..exporters.base_exporter import Exporter
from ..extractors.igrav import IGravExtractor


class IGrav(IGravExtractor, IGravCorrection, Exporter):
    """
    Combined class for iGrav data extraction, corrections and export.
    """


__all__ = ["IGrav"]
