"""Compatibility package root for HainaOCR-NativePixel.

The canonical package is ``hainaocr_nativepixel`` under the subdirectory
of the same name. This root __init__.py re-exports the public API so that
adding the *parent* directory to ``PYTHONPATH`` works.
"""

from hainaocr_nativepixel.hainaocr_nativepixel.configuration import HainaOCRNativePixelConfig
from hainaocr_nativepixel.hainaocr_nativepixel.modeling import (
    HainaOCRNativePixelForConditionalGeneration,
    HainaOCRNativePixelModel,
)
from hainaocr_nativepixel.hainaocr_nativepixel.processing import (
    HainaOCRNativePixelImageProcessor,
    HainaOCRNativePixelProcessor,
)

__all__ = [
    "HainaOCRNativePixelConfig",
    "HainaOCRNativePixelForConditionalGeneration",
    "HainaOCRNativePixelImageProcessor",
    "HainaOCRNativePixelModel",
    "HainaOCRNativePixelProcessor",
]
