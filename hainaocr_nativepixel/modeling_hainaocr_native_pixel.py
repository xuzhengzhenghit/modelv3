"""Backward-compatible model import path.

New code should import from ``hainaocr_nativepixel.modeling``.
This wrapper keeps old AutoModel / Swift plugin paths working.
"""

from hainaocr_nativepixel.modeling import (
    HainaOCRNativePixelCausalLMOutputWithPast,
    HainaOCRNativePixelForConditionalGeneration,
    HainaOCRNativePixelModel,
    HainaOCRNativePixelModelOutput,
    HainaOCRNativePixelPreTrainedModel,
    PositionEmbedding2D,
    SimplePatchEmbedding,
)

__all__ = [
    "HainaOCRNativePixelCausalLMOutputWithPast",
    "HainaOCRNativePixelForConditionalGeneration",
    "HainaOCRNativePixelModel",
    "HainaOCRNativePixelModelOutput",
    "HainaOCRNativePixelPreTrainedModel",
    "PositionEmbedding2D",
    "SimplePatchEmbedding",
]
