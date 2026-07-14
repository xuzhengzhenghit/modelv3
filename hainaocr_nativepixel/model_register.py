"""MS-Swift registration for HainaOCR-NativePixel."""

from __future__ import annotations

import os
from pathlib import Path
import torch
from typing import Any, Dict, List, Literal

from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

from configuration_hainaocr_native_pixel import HainaOCRNativePixelConfig
from modeling_hainaocr_native_pixel import HainaOCRNativePixelForConditionalGeneration
from processing_hainaocr_native_pixel import (
    HainaOCRNativePixelImageProcessor,
    HainaOCRNativePixelProcessor,
)

AutoConfig.register("hainaocr_native_pixel", HainaOCRNativePixelConfig, exist_ok=True)
AutoModelForCausalLM.register(
    HainaOCRNativePixelConfig,
    HainaOCRNativePixelForConditionalGeneration,
    exist_ok=True,
)
AutoProcessor.register(HainaOCRNativePixelConfig, HainaOCRNativePixelProcessor, exist_ok=True)

try:
    from swift.model import (
        Model,
        ModelGroup,
        ModelLoader,
        ModelMeta,
        MultiModelKeys,
        register_model,
        register_model_arch,
    )
    from swift.template.base import Template
    from swift.template.register import TemplateMeta, register_template
    from swift.template.template_inputs import StdTemplateInputs
    from swift.template.utils import Context, findall
    from swift.utils import Processor, to_float_dtype
except Exception:
    ModelLoader = None


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_LLM_DIR = Path("/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b")
MODEL_TYPE = "hainaocr_native_pixel"
TEMPLATE_TYPE = "hainaocr_native_pixel"
IMAGE_TOKEN_ID = 151655


def _resolve_llm_dir(config: HainaOCRNativePixelConfig) -> str:
    candidates = [
        getattr(config, "llm_model_name", None),
        getattr(config, "source_llm_model_name", None),
        str(DEFAULT_LLM_DIR),
    ]
    for candidate in candidates:
        if candidate and Path(str(candidate)).exists():
            return str(candidate)
    raise FileNotFoundError(
        "Cannot find Qwen LLM weights. Tried: "
        + ", ".join(str(c) for c in candidates if c)
    )


def _torch_dtype_from(config: PretrainedConfig, model_kwargs: Dict[str, Any], kwargs: Dict[str, Any]):
    return (
        getattr(config, "torch_dtype", None)
        or getattr(config, "dtype", None)
        or model_kwargs.get("torch_dtype")
        or model_kwargs.get("dtype")
        or kwargs.get("torch_dtype")
        or kwargs.get("dtype")
    )


def _bool_from(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _first_present(name: str, *sources: Dict[str, Any]):
    for source in sources:
        if source is not None and name in source and source[name] is not None:
            return source[name]
    return None


def _apply_freeze_overrides(config: HainaOCRNativePixelConfig, model_kwargs: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, bool]:
    """Make Swift freeze flags explicit before the model is constructed."""
    freeze_llm = _first_present("freeze_llm", model_kwargs, kwargs)
    if freeze_llm is not None:
        config.freeze_llm = _bool_from(freeze_llm)

    freeze_vit = _first_present("freeze_vit", model_kwargs, kwargs)
    if freeze_vit is not None:
        config.freeze_patch_embed = _bool_from(freeze_vit)

    freeze_aligner = _first_present("freeze_aligner", model_kwargs, kwargs)
    return {"freeze_aligner": _bool_from(freeze_aligner) if freeze_aligner is not None else False}


def _set_requires_grad(module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


if ModelLoader is not None:

    class HainaOCRNativePixelTemplate(Template):
        image_token_id = IMAGE_TOKEN_ID
        placeholder_tokens = ["<|image_pad|>"]
        use_model = True
        support_padding_free = True
        norm_bbox = "none"

        def replace_tag(
            self,
            media_type: Literal["image", "video", "audio"],
            index: int,
            inputs: StdTemplateInputs,
        ) -> List[Context]:
            assert media_type == "image"
            return ["<|vision_start|><|image_pad|><|vision_end|>"]

        def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
            # Preprocessed data: images[0] has _pp → skip image loading
            if (inputs.images and isinstance(inputs.images[0], dict)
                    and inputs.images[0].get("_pp")):
                return
            super()._preprocess_inputs(inputs)

        def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
            encoded = super()._encode(inputs)
            images = inputs.images
            if not images:
                return encoded

            # Preprocessed Arrow: check various places _pp might survive
            _pp = None
            # 1. Via StdTemplateInputs.kwargs (from messages construction)
            if hasattr(inputs, 'kwargs') and inputs.kwargs:
                _pp = inputs.kwargs.get('_pp')
            # 2. In images dict directly
            if _pp is None and isinstance(images[0], dict):
                _pp = images[0].get('_pp')
            # 3. Old format compat
            if _pp is None:
                _pp = getattr(inputs, '_pp', None)

            if _pp is not None and isinstance(_pp, dict):
                import numpy as np
                shape = [int(x) for x in _pp["shape"].split(",")]
                pv = np.frombuffer(_pp["bytes"], dtype=np.float16).reshape(shape)
                pv = torch.from_numpy(pv.astype(np.float32))
                grid = [int(x) for x in _pp["grid"].split(",")]
                image_inputs = {
                    "pixel_values": pv,
                    "image_grid_thw": torch.tensor([grid]),
                }
            elif isinstance(images[0], dict) and images[0].get("bytes") and not images[0].get("bytes"):
                # Empty bytes dict → dummy image from preprocessor → skip
                return encoded
            else:
                image_inputs = self.processor.image_processor(images=images, return_tensors="pt")

            model_dtype = getattr(getattr(self, "model_info", None), "torch_dtype", None)
            if model_dtype is not None:
                image_inputs = to_float_dtype(image_inputs, model_dtype)

            input_ids = encoded["input_ids"]
            labels = encoded["labels"]
            loss_scale = encoded.get("loss_scale", None)
            image_grid_thw = image_inputs["image_grid_thw"]
            idx_list = findall(input_ids, self.image_token_id)

            def _get_new_tokens(i: int) -> List[int]:
                grid = image_grid_thw[i]
                if hasattr(grid, "tolist"):
                    grid = grid.tolist()
                grid_t, grid_h, grid_w = grid
                token_len = int(grid_t) * int(grid_h) * int(grid_w)
                return [self.image_token_id] * token_len

            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids,
                labels,
                loss_scale,
                idx_list,
                _get_new_tokens,
            )
            encoded["input_ids"] = input_ids
            encoded["labels"] = labels
            encoded["loss_scale"] = loss_scale
            encoded["pixel_values"] = image_inputs["pixel_values"]
            encoded["image_grid_thw"] = image_grid_thw
            return encoded


    class HainaOCRNativePixelLoader(ModelLoader):

        def get_config(self, model_dir: str) -> PretrainedConfig:
            config = HainaOCRNativePixelConfig.from_pretrained(model_dir, trust_remote_code=True)
            llm_dir = _resolve_llm_dir(config)
            config.llm_model_name = llm_dir
            config.source_llm_model_name = llm_dir
            return config

        def get_processor(self, model_dir: str, config: PretrainedConfig, **kwargs) -> Processor:
            llm_dir = _resolve_llm_dir(config)
            tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
            image_processor = HainaOCRNativePixelImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
            return HainaOCRNativePixelProcessor(image_processor=image_processor, tokenizer=tokenizer)

        def get_model(self, model_dir, config, processor, model_kwargs, **kwargs) -> PreTrainedModel:
            torch_dtype = _torch_dtype_from(config, model_kwargs, kwargs)
            llm_dir = _resolve_llm_dir(config)
            freeze_flags = _apply_freeze_overrides(config, model_kwargs, kwargs)
            model = HainaOCRNativePixelForConditionalGeneration(config)
            model.load_pretrained_components(llm_model_name=llm_dir, dtype=torch_dtype)

            # Load pre-trained CNN weights if provided
            cnn_weights = (
                _first_present("cnn_weights", model_kwargs, kwargs)
                or os.environ.get("HAINA_CNN_WEIGHTS")
            )
            if cnn_weights:
                from safetensors.torch import load_file as _load_st
                if not os.path.exists(cnn_weights):
                    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                        print(f"[CNN] WARNING: cnn_weights not found at {cnn_weights}, skipping")
                else:
                    cnn_sd = _load_st(cnn_weights)
                    model.load_state_dict(cnn_sd, strict=False)
                    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                        print(f"[CNN] loaded {len(cnn_sd)} keys from {cnn_weights}")

            if freeze_flags["freeze_aligner"]:
                _set_requires_grad(model.model.pos_embed_2d, False)
            if torch_dtype is not None:
                model = model.to(dtype=torch_dtype)
            return model


    def register_hainaocr_native_pixel_model() -> None:
        register_model_arch(
            MultiModelKeys(
                MODEL_TYPE,
                language_model=["model.qwen3", "lm_head"],
                vision_tower=["model.vision_encoder"],
                aligner=["model.pos_embed_2d"],
                generator=[],
            ),
            exist_ok=True,
        )

        register_model(
            ModelMeta(
                model_type=MODEL_TYPE,
                model_groups=[ModelGroup([Model("local/hainaocr-nativepixel", str(MODEL_DIR))])],
                loader=HainaOCRNativePixelLoader,
                template=TEMPLATE_TYPE,
                is_multimodal=True,
                model_arch=MODEL_TYPE,
                architectures=["HainaOCRNativePixelForConditionalGeneration"],
                requires=["transformers>=4.57", "torch>=2.5.0"],
                tags=["vision", "ocr", "multimodal", "native-pixel"],
            ),
            exist_ok=True,
        )


    def register_hainaocr_native_pixel_template() -> None:
        register_template(
            TemplateMeta(
                template_type=TEMPLATE_TYPE,
                prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
                prompt=[
                    "<|im_start|>user\n{{QUERY}}<|im_end|>\n",
                    "<|im_start|>assistant\n",
                ],
                chat_sep=["<|im_end|>\n"],
                suffix=["<|im_end|>"],
                template_cls=HainaOCRNativePixelTemplate,
                default_system="You are a helpful assistant.",
            ),
            exist_ok=True,
        )


    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    register_hainaocr_native_pixel_model()
    register_hainaocr_native_pixel_template()
