"""Encoder-free HainaOCRNativePixel model with Apple-style NativePixel front-end."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RMSNorm
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration import HainaOCRNativePixelConfig


def _flash_attn_forward(self, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, **kwargs):
    """C550/MetaX flash-attn adapter — standard (ignores attention_mask)."""
    from flash_attn import flash_attn_func

    q, k, v = [x.transpose(1, 2).contiguous() for x in (query, key, value)]
    out = flash_attn_func(q, k, v, dropout_p=dropout, causal=True)
    return out.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS["flash_attn_c550"] = _flash_attn_forward

# ── Varlen attention (supports packing with cu_seqlens) ──────────────

_flash_attn_cu_seqlens: Optional[torch.Tensor] = None
"""Module-level state set before model forward when using varlen packing."""


def _flash_attn_varlen_forward(self, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, **kwargs):
    """C550/MetaX flash-attn adapter — varlen variant.

    When ``_flash_attn_cu_seqlens`` is set (by the training loop before model
    forward), uses ``flash_attn_varlen_func`` for correct block-diagonal-causal
    masking across packed sub-sequences.

    Falls back to regular ``flash_attn_func`` when cu_seqlens is None (e.g.
    during inference or non-packed training).
    """
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    global _flash_attn_cu_seqlens

    if _flash_attn_cu_seqlens is not None and query.shape[0] == 1:
        # Packed mode: [1, nhead, T, head_dim] → [T, nhead, head_dim]
        q, k, v = [x.squeeze(0).transpose(0, 1).contiguous() for x in (query, key, value)]
        cu = _flash_attn_cu_seqlens
        max_seq = max(cu[i + 1] - cu[i] for i in range(len(cu) - 1)).item()
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_seq, max_seqlen_k=max_seq,
            dropout_p=dropout, causal=True,
        )
    else:
        q, k, v = [x.transpose(1, 2).contiguous() for x in (query, key, value)]
        out = flash_attn_func(q, k, v, dropout_p=dropout, causal=True)

    return out.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS["flash_attn_c550_varlen"] = _flash_attn_varlen_forward

try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

    _HAS_LIGER = True
except Exception:
    LigerFusedLinearCrossEntropyFunction = None
    _HAS_LIGER = False


@dataclass
class HainaOCRNativePixelModelOutput(BaseModelOutputWithPast):
    expanded_labels: Optional[torch.LongTensor] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class HainaOCRNativePixelCausalLMOutputWithPast(CausalLMOutputWithPast):
    image_hidden_states: Optional[torch.FloatTensor] = None
    expanded_labels: Optional[torch.LongTensor] = None
    loss_ntp: Optional[torch.FloatTensor] = None


class SimplePatchEmbedding(nn.Module):
    """Apple/NEO-style Conv2d patchifier + RMSNorm.

    Conv1(stride=patch_size) -> GELU -> Conv2(stride=spatial_merge_size) -> RMSNorm.
    With patch_size=16 and spatial_merge_size=2, each visual token covers 32x32 pixels.
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3, hidden_size: int = 2048, spatial_merge_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.effective_patch_size = patch_size * spatial_merge_size
        self.conv1 = nn.Conv2d(in_channels, hidden_size // 2, kernel_size=patch_size, stride=patch_size, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            hidden_size // 2,
            hidden_size,
            kernel_size=spatial_merge_size,
            stride=spatial_merge_size,
            bias=True,
        )
        self.norm = Qwen3RMSNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.conv1(pixel_values)
        x = self.act(x)
        x = self.conv2(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x, (h, w)


class PositionEmbedding2D(nn.Module):
    """Resolution-independent 2D coordinate MLP positional embedding."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.arange(grid_h, device=device, dtype=torch.float32) / max(grid_h - 1, 1)
        x = torch.arange(grid_w, device=device, dtype=torch.float32) / max(grid_w - 1, 1)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).to(dtype)
        return self.mlp(coords)


class HainaOCRNativePixelPreTrainedModel(PreTrainedModel):
    config_class = HainaOCRNativePixelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _supports_flash_attn_2 = True
    _tied_weights_keys = ["lm_head.weight"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, Qwen3RMSNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_vision_boundary_embeds(self):
        """Initialize learnable vision boundary embeddings from original Qwen3 embed_tokens."""
        emb = self.qwen3.embed_tokens.weight
        with torch.no_grad():
            self.vision_start_embed.copy_(emb[self.config.vision_start_token_id])
            self.vision_end_embed.copy_(emb[self.config.vision_end_token_id])


class HainaOCRNativePixelModel(HainaOCRNativePixelPreTrainedModel):
    """NativePixel backbone: Apple patch embed + 2D MLP pos embed + Qwen3Model."""

    def __init__(self, config: HainaOCRNativePixelConfig):
        super().__init__(config)
        self.vision_encoder = SimplePatchEmbedding(
            patch_size=config.patch_size,
            in_channels=3,
            hidden_size=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.pos_embed_2d = PositionEmbedding2D(config.hidden_size)
        self.qwen3 = Qwen3Model(config)
        for layer in self.qwen3.layers:
            layer.self_attn.config._attn_implementation = getattr(config, "_attn_implementation", "flash_attn_c550")
        self.image_token_id = config.image_token_index
        # Learnable vision boundary embeddings (independent of frozen embed_tokens)
        self.vision_start_embed = nn.Parameter(torch.empty(config.hidden_size))
        self.vision_end_embed = nn.Parameter(torch.empty(config.hidden_size))
        self.post_init()

    def get_input_embeddings(self):
        return self.qwen3.embed_tokens

    def set_input_embeddings(self, value):
        self.qwen3.embed_tokens = value

    @staticmethod
    def _normalize_image_grid_thw(image_grid_thw, num_images: int, fallback_grid: Tuple[int, int]):
        if image_grid_thw is None:
            grid_h, grid_w = fallback_grid
            return [(1, grid_h, grid_w) for _ in range(num_images)]
        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.detach().cpu().tolist()
        normalized = []
        for grid in image_grid_thw:
            if isinstance(grid, torch.Tensor):
                grid = grid.detach().cpu().tolist()
            if len(grid) == 2:
                normalized.append((1, int(grid[0]), int(grid[1])))
            else:
                normalized.append((int(grid[0]), int(grid[1]), int(grid[2])))
        return normalized

    def _encode_images(self, pixel_values: torch.Tensor, image_grid_thw=None) -> List[torch.Tensor]:
        dtype = self.get_input_embeddings().weight.dtype
        device = self.get_input_embeddings().weight.device
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        grids_for_patch_mode = None
        if image_grid_thw is not None:
            grids_for_patch_mode = self._normalize_image_grid_thw(image_grid_thw, 0, (1, 1))
            expected_tokens = sum(t * h * w for t, h, w in grids_for_patch_mode)
            if pixel_values.ndim == 4 and expected_tokens == pixel_values.shape[0]:
                image_embeds, (patch_grid_h, patch_grid_w) = self.vision_encoder(pixel_values)
                if patch_grid_h != 1 or patch_grid_w != 1:
                    raise ValueError(
                        "Qwen-style patch sequence input expects each visual token to encode to one grid cell; "
                        f"got {(patch_grid_h, patch_grid_w)}."
                    )
                image_embeds = image_embeds.squeeze(1)
                encoded_images: List[torch.Tensor] = []
                offset = 0
                for grid_t, grid_h, grid_w in grids_for_patch_mode:
                    token_count = int(grid_t) * int(grid_h) * int(grid_w)
                    image_tokens = image_embeds[offset : offset + token_count]
                    pos = self.pos_embed_2d(grid_h, grid_w, image_tokens.device, image_tokens.dtype)
                    if grid_t == 1:
                        image_tokens = image_tokens + pos
                    else:
                        image_tokens = image_tokens + pos.repeat(grid_t, 1)
                    encoded_images.append(image_tokens)
                    offset += token_count
                return encoded_images

        image_embeds, (grid_h, grid_w) = self.vision_encoder(pixel_values)
        pos = self.pos_embed_2d(grid_h, grid_w, image_embeds.device, image_embeds.dtype)
        image_embeds = image_embeds + pos.unsqueeze(0)

        grids = self._normalize_image_grid_thw(image_grid_thw, image_embeds.shape[0], (grid_h, grid_w))
        encoded_images: List[torch.Tensor] = []
        for i, (_, real_h, real_w) in enumerate(grids):
            if real_h > grid_h or real_w > grid_w:
                raise ValueError(
                    f"image_grid_thw[{i}]={grids[i]} exceeds encoded grid {(grid_h, grid_w)}."
                )
            image_2d = image_embeds[i].reshape(grid_h, grid_w, -1)
            encoded_images.append(image_2d[:real_h, :real_w, :].reshape(real_h * real_w, -1))
        return encoded_images

    def _replace_image_placeholders(self, inputs_embeds, input_ids, image_embeds):
        # ── Hard assertion 1: total placeholders must match total visual tokens ──
        total_placeholders = sum(
            (input_ids[i] == self.config.image_token_index).sum().item()
            for i in range(input_ids.shape[0])
        )
        total_visual = sum(x.shape[0] for x in image_embeds)
        if total_placeholders != total_visual:
            raise ValueError(
                f"Visual token mismatch BEFORE replacement: "
                f"placeholders={total_placeholders}, visual={total_visual}"
            )

        inputs_embeds = inputs_embeds.clone()

        # ── Replace VISION_START / VISION_END with learnable boundary embeddings ──
        vs_id = self.config.vision_start_token_id
        ve_id = self.config.vision_end_token_id
        for batch_index in range(input_ids.shape[0]):
            vs_mask = input_ids[batch_index] == vs_id
            ve_mask = input_ids[batch_index] == ve_id
            if vs_mask.any():
                inputs_embeds[batch_index, vs_mask] = self.vision_start_embed.to(inputs_embeds.dtype)
            if ve_mask.any():
                inputs_embeds[batch_index, ve_mask] = self.vision_end_embed.to(inputs_embeds.dtype)

        # ── Replace IMAGE_PAD with visual features ──
        image_index = 0
        for batch_index in range(input_ids.shape[0]):
            image_positions = torch.nonzero(
                input_ids[batch_index] == self.config.image_token_index,
                as_tuple=False,
            ).flatten()

            # ── Hard assertion 2: placeholder positions must be contiguous ──
            if image_positions.numel() > 1:
                diffs = image_positions[1:] - image_positions[:-1]
                if not torch.all(diffs == 1):
                    raise ValueError(
                        f"Image placeholder positions are not contiguous at "
                        f"batch index {batch_index}: "
                        f"{image_positions[:20].detach().cpu().tolist()}"
                    )
            offset = 0
            while offset < image_positions.numel():
                if image_index >= len(image_embeds):
                    raise ValueError(
                        f"More image placeholders than encoded images at batch index {batch_index}."
                    )
                num_image_tokens = int(image_embeds[image_index].shape[0])
                positions = image_positions[offset : offset + num_image_tokens]
                if positions.numel() != num_image_tokens:
                    raise ValueError(
                        f"Image placeholder run is shorter than image embeddings "
                        f"({positions.numel()} vs {num_image_tokens}) for image {image_index}."
                    )
                inputs_embeds[batch_index, positions] = image_embeds[image_index].to(inputs_embeds.dtype)
                offset += num_image_tokens
                image_index += 1
            if offset != image_positions.numel():
                raise ValueError(
                    f"Unconsumed image placeholders for batch index {batch_index}."
                )
        if image_index != len(image_embeds):
            raise ValueError(f"Encoded image count ({len(image_embeds)}) does not match placeholders ({image_index}).")
        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[Union[List[Tuple[int, int, int]], torch.Tensor]] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_hidden_states = None
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids must be provided when inputs_embeds is None.")
            text_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                # Guard: skip if input_ids have no image tokens (e.g. DataParallel
                # scatters pixel_values from image samples to text-only replicas)
                has_image_tokens = (input_ids == self.config.image_token_index).any()
                if has_image_tokens:
                    image_hidden_states = self._encode_images(pixel_values, image_grid_thw)
                    inputs_embeds = self._replace_image_placeholders(text_embeds, input_ids, image_hidden_states)
                else:
                    inputs_embeds = text_embeds
            else:
                inputs_embeds = text_embeds

        outputs = self.qwen3(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        if not return_dict:
            return outputs
        return HainaOCRNativePixelModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            expanded_labels=labels,
            image_hidden_states=image_hidden_states,
        )


class HainaOCRNativePixelForConditionalGeneration(HainaOCRNativePixelPreTrainedModel, GenerationMixin):
    """Causal LM wrapper with in-model Qwen3Model and lm_head."""

    def __init__(self, config: HainaOCRNativePixelConfig):
        config._attn_implementation = getattr(config, "_attn_implementation", "flash_attn_c550")
        super().__init__(config)
        self.model = HainaOCRNativePixelModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_token_index = config.image_token_index
        self.post_init()
        for layer in self.model.qwen3.layers:
            layer.self_attn.config._attn_implementation = config._attn_implementation
        if config.freeze_patch_embed:
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False
        if config.freeze_llm:
            for param in self.model.qwen3.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _select_qwen3_layers_state_dict(self, source_state_dict):
        selected = {}
        layer_indices = list(getattr(self.config, "qwen3_layer_indices", range(self.config.num_hidden_layers)))
        source_to_target = {source_idx: target_idx for target_idx, source_idx in enumerate(layer_indices)}
        for key, value in source_state_dict.items():
            if not key.startswith("layers."):
                selected[key] = value
                continue
            parts = key.split(".", 2)
            if len(parts) < 3:
                continue
            source_idx = int(parts[1])
            target_idx = source_to_target.get(source_idx)
            if target_idx is None:
                continue
            selected[f"layers.{target_idx}.{parts[2]}"] = value
        return selected

    def load_pretrained_components(self, llm_model_name: str = None, device_map: str = None, dtype=None, torch_dtype=None):
        dtype = dtype if dtype is not None else torch_dtype
        llm_name = llm_model_name or self.config.llm_model_name or self.config.source_llm_model_name
        if not llm_name:
            raise ValueError("llm_model_name is required to load Qwen3 weights.")
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=True,
        )
        qwen3_state = self._select_qwen3_layers_state_dict(llm.model.state_dict())
        self.model.qwen3.load_state_dict(qwen3_state, strict=True)
        self.lm_head.load_state_dict(llm.lm_head.state_dict(), strict=True)
        del llm
        if dtype is not None:
            self.to(dtype=dtype)
        # Initialize learnable vision boundary embeddings from the loaded Qwen3 embed_tokens
        self.model._init_vision_boundary_embeds()
        print("Loaded pretrained components:")
        print(f"  - Qwen3Model/lm_head state_dict: {llm_name}")
        print(f"  - Qwen3 layers: {self.config.qwen3_layer_indices} -> 0..{self.config.num_hidden_layers - 1}")
        print("  - Vision: Apple SimplePatchEmbedding + 2D MLP position embedding + learnable boundary embeds")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if "dtype" not in kwargs and "torch_dtype" in kwargs:
            kwargs["dtype"] = kwargs["torch_dtype"]
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(config_path):
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        config = HainaOCRNativePixelConfig.from_pretrained(pretrained_model_name_or_path)
        dtype = kwargs.get("dtype", None)
        model = cls(config)
        model_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.exists(model_path):
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[NativePixel] missing keys while loading checkpoint: {missing[:10]} ({len(missing)} total)")
            if unexpected:
                print(f"[NativePixel] unexpected keys while loading checkpoint: {unexpected[:10]} ({len(unexpected)} total)")
        elif config.llm_model_name or config.source_llm_model_name:
            model.load_pretrained_components(dtype=dtype)
        else:
            raise ValueError("No model.safetensors or llm_model_name available.")
        if dtype is not None:
            model = model.to(dtype=dtype)
        return model

    def _sync_export_config(self):
        self.config.torch_dtype = str(next(self.parameters()).dtype).replace("torch.", "")

    def save_pretrained(self, save_directory, *args, **kwargs):
        self._sync_export_config()
        return super().save_pretrained(save_directory, *args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )
        # Only clear pixel_values when cache actually has content.
        # super().prepare_inputs_for_generation() may return an empty
        # DynamicCache on step 0, which is not None but has no stored keys.
        cache_has_values = (
            past_key_values is not None
            and getattr(past_key_values, "get_seq_length", lambda: 0)() > 0
        )
        if cache_has_values:
            pixel_values = None
            image_grid_thw = None
        model_inputs.update({"pixel_values": pixel_values, "image_grid_thw": image_grid_thw})
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[Union[List[Tuple[int, int, int]], torch.Tensor]] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        loss_ntp = None
        if labels is not None:
            if self.config.use_liger_ce and _HAS_LIGER:
                hs = hidden_states.reshape(-1, hidden_states.shape[-1])
                flat_labels = labels.reshape(-1)
                loss_tuple = LigerFusedLinearCrossEntropyFunction.apply(
                    hs[:-1], self.lm_head.weight, flat_labels[1:],
                    None, None, -100, 0.0, 0.0, "mean", None, False,
                )
                loss_ntp = loss_tuple[0]
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_ntp = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                )
            loss = loss_ntp

        if not return_dict:
            if loss is not None:
                return (loss, logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)

        return HainaOCRNativePixelCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            expanded_labels=outputs.expanded_labels,
            loss_ntp=loss_ntp,
        )

    @torch.no_grad()
    def generate(self, pixel_values=None, input_ids=None, attention_mask=None, image_grid_thw=None, **generate_kwargs):
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **generate_kwargs,
        )
