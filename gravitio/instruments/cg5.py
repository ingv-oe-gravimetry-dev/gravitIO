from ..corrections.scintrex_cgx import CGXCorrection
from ..exporters.base_exporter import Exporter
from ..extractors.scintrex_cg5 import CG5Extractor


class CG5(CG5Extractor, Exporter, CGXCorrection):
    """
    Combined class for Scintrex CG-X data extraction, corrections and export.
    """

    def __init__(self) -> None:
        super().__init__()
        self.grav_col_name: str = "grav"


__all__ = ["CG5"]
