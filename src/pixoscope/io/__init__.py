"""Backends de lecture d'image : interface commune et sélection automatique."""

from pixoscope.io.backend_base import ImageBackend, ImageHandle, PyramidLevelInfo
from pixoscope.io.backend_registry import open_image

__all__ = ["ImageBackend", "ImageHandle", "PyramidLevelInfo", "open_image"]
