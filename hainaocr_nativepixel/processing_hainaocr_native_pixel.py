"""Backward-compatible processor import path.

New code should import from ``hainaocr_nativepixel.processing``.
This wrapper keeps old AutoProcessor / Swift plugin paths working.
"""

from hainaocr_nativepixel.processing import (
    HainaOCRNativePixelImageProcessor,
    HainaOCRNativePixelProcessor,
    load_tokenizer_with_fixes,
)

__all__ = [
    "HainaOCRNativePixelImageProcessor",
    "HainaOCRNativePixelProcessor",
    "load_tokenizer_with_fixes",
]
