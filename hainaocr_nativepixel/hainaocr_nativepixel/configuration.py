"""HainaOCRNativePixel configuration.

This config follows the hainaocr-apple NativePixel design: it subclasses
Qwen3Config so the text backbone is an in-model Qwen3Model instead of an
external AutoModelForCausalLM wrapper.
"""

from typing import List, Optional

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


QWEN3_0_6B_DEFAULTS = {
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "source_num_hidden_layers": 28,
    "qwen3_layer_indices": list(range(28)),
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 32647,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
}


def _default_if_none(value, default):
    return default if value is None else value


class HainaOCRNativePixelConfig(Qwen3Config):
    """Qwen3-native config with lightweight raw-pixel patch embedding fields."""

    model_type = "hainaocr_native_pixel"

    def __init__(
        self,
        llm_model_name: Optional[str] = "/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b",
        source_llm_model_name: Optional[str] = None,
        image_size: int = 384,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        image_token_index: int = 7,
        vision_start_token_id: int = 5,
        vision_end_token_id: int = 6,
        vision_pad_token_id: int = 0,
        image_start_token_index=None,
        image_end_token_index=None,
        min_pixels: int = 224 * 224,
        max_pixels: int = 3_200_000,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        freeze_llm: bool = False,
        freeze_patch_embed: bool = False,
        pre_buffer_layers: int = 0,
        use_liger_ce: bool = True,
        source_num_hidden_layers: int = 28,
        qwen3_layer_indices: Optional[List[int]] = None,
        # Qwen3-0.6B defaults.
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        vocab_size: int = 32647,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        tie_word_embeddings: bool = True,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # Compatibility with earlier configs.
        visual_hidden_size = kwargs.pop("visual_hidden_size", None)
        kwargs.pop("visual_intermediate_size", None)
        kwargs.pop("use_2d_sincos_pos", None)
        kwargs.pop("max_image_size", None)
        kwargs.pop("llm_config", None)
        kwargs.pop("text_config", None)

        if hidden_size is None and visual_hidden_size is not None:
            hidden_size = visual_hidden_size

        hidden_size = _default_if_none(hidden_size, QWEN3_0_6B_DEFAULTS["hidden_size"])
        intermediate_size = _default_if_none(intermediate_size, QWEN3_0_6B_DEFAULTS["intermediate_size"])
        num_hidden_layers = _default_if_none(num_hidden_layers, QWEN3_0_6B_DEFAULTS["num_hidden_layers"])
        num_attention_heads = _default_if_none(num_attention_heads, QWEN3_0_6B_DEFAULTS["num_attention_heads"])
        num_key_value_heads = _default_if_none(num_key_value_heads, QWEN3_0_6B_DEFAULTS["num_key_value_heads"])
        head_dim = _default_if_none(head_dim, QWEN3_0_6B_DEFAULTS["head_dim"])
        vocab_size = _default_if_none(vocab_size, QWEN3_0_6B_DEFAULTS["vocab_size"])
        hidden_act = _default_if_none(hidden_act, QWEN3_0_6B_DEFAULTS["hidden_act"])
        rms_norm_eps = _default_if_none(rms_norm_eps, QWEN3_0_6B_DEFAULTS["rms_norm_eps"])
        rope_theta = _default_if_none(rope_theta, QWEN3_0_6B_DEFAULTS["rope_theta"])
        tie_word_embeddings = _default_if_none(tie_word_embeddings, QWEN3_0_6B_DEFAULTS["tie_word_embeddings"])
        patch_size = _default_if_none(patch_size, 16)
        spatial_merge_size = _default_if_none(spatial_merge_size, 2)
        source_num_hidden_layers = _default_if_none(
            source_num_hidden_layers,
            QWEN3_0_6B_DEFAULTS["source_num_hidden_layers"],
        )
        if qwen3_layer_indices is None:
            qwen3_layer_indices = list(QWEN3_0_6B_DEFAULTS["qwen3_layer_indices"])

        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.llm_model_name = llm_model_name
        self.source_llm_model_name = source_llm_model_name or llm_model_name

        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.effective_patch_size = patch_size * spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)

        self.image_token_index = image_token_index
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.vision_pad_token_id = vision_pad_token_id
        self.image_start_token_index = image_start_token_index
        self.image_end_token_index = image_end_token_index

        self.freeze_llm = freeze_llm
        self.freeze_patch_embed = freeze_patch_embed
        self.pre_buffer_layers = pre_buffer_layers
        self.use_liger_ce = use_liger_ce
        self.source_num_hidden_layers = source_num_hidden_layers
        self.qwen3_layer_indices = list(qwen3_layer_indices)
        self.torch_dtype = torch_dtype

        if len(self.qwen3_layer_indices) != self.num_hidden_layers:
            raise ValueError(
                "qwen3_layer_indices length must equal num_hidden_layers: "
                f"{len(self.qwen3_layer_indices)} vs {self.num_hidden_layers}."
            )
        if any(i < 0 or i >= self.source_num_hidden_layers for i in self.qwen3_layer_indices):
            raise ValueError(
                "qwen3_layer_indices must be within source_num_hidden_layers: "
                f"indices={self.qwen3_layer_indices}, source={self.source_num_hidden_layers}."
            )
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive.")
        if self.min_pixels <= 0 or self.max_pixels <= 0:
            raise ValueError("min_pixels and max_pixels must be positive.")
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels cannot exceed max_pixels.")
