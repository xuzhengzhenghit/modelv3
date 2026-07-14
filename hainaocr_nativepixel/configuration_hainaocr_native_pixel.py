"""Backward-compatible config import path.

New code should import from ``hainaocr_nativepixel.configuration``.
This wrapper keeps old AutoConfig / Swift plugin paths working.
"""

from hainaocr_nativepixel.configuration import HainaOCRNativePixelConfig

__all__ = ["HainaOCRNativePixelConfig"]
