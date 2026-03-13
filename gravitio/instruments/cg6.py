from ..corrections.scintrex_cgx import CGXCorrection
from ..exporters.base_exporter import Exporter
from ..extractors.scintrex_cg6 import CG6Extractor


class CG6(CG6Extractor, CGXCorrection, Exporter):
    """
    Combined class for Scintrex CG-X data extraction, corrections and export.
    """

    def __init__(self) -> None:
        super().__init__()
        self.grav_col_name: str = "corrgrav"


__all__ = ["CG6"]
