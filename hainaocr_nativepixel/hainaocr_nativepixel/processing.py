"""HainaOCRNativePixel processor."""

import math
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoTokenizer, ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import ImageInput, PILImageResampling, make_list_of_images, valid_images


def load_tokenizer_with_fixes(pretrained_model_name_or_path, **kwargs):
    try:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)


class HainaOCRNativePixelImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        image_mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        image_std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        min_pixels: int = 224 * 224,
        max_pixels: int = 3_200_000,
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.effective_patch_size = patch_size * spatial_merge_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.resample = resample

    def smart_resize(self, height: int, width: int) -> Tuple[int, int]:
        factor = self.effective_patch_size
        resized_height = max(factor, round(height / factor) * factor)
        resized_width = max(factor, round(width / factor) * factor)
        if resized_height * resized_width > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            resized_height = max(factor, math.floor(height / beta / factor) * factor)
            resized_width = max(factor, math.floor(width / beta / factor) * factor)
        elif resized_height * resized_width < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            resized_height = max(factor, math.ceil(height * beta / factor) * factor)
            resized_width = max(factor, math.ceil(width * beta / factor) * factor)
        return resized_width, resized_height

    def resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        new_width, new_height = self.smart_resize(height, width)
        return image.resize((new_width, new_height), self.resample)

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        import numpy as np
        np_image = np.array(image).astype("float32") / 255.0
        tensor = torch.from_numpy(np_image).permute(2, 0, 1)
        mean = torch.tensor(self.image_mean).view(3, 1, 1)
        std = torch.tensor(self.image_std).view(3, 1, 1)
        return (tensor - mean) / std

    def _to_patch_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert one image tensor into fixed-size visual patches.

        This follows the Qwen-style dynamic-resolution contract used by the
        rest of this project: different images may produce different numbers of
        visual tokens, but every token has the same tensor shape, so a batch can
        concatenate them safely.
        """

        patch = self.effective_patch_size
        c, h, w = tensor.shape
        if h % patch != 0 or w % patch != 0:
            raise ValueError(f"Image size {(h, w)} must be divisible by patch size {patch}.")
        return (
            tensor.reshape(c, h // patch, patch, w // patch, patch)
            .permute(1, 3, 0, 2, 4)
            .reshape((h // patch) * (w // patch), c, patch, patch)
            .contiguous()
        )

    def _pad_image_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Pad resized image tensors so Conv2d patch embedding can run on GPU.

        The model crops visual tokens back to each image's real grid using
        ``image_grid_thw``. Padding with zero in normalized space corresponds to
        a neutral 0.5 RGB value for the current mean/std.
        """

        max_h = max(tensor.shape[1] for tensor in tensors)
        max_w = max(tensor.shape[2] for tensor in tensors)
        batch = tensors[0].new_zeros((len(tensors), tensors[0].shape[0], max_h, max_w))
        for i, tensor in enumerate(tensors):
            _, h, w = tensor.shape
            batch[i, :, :h, :w] = tensor
        return batch

    def preprocess(self, images: ImageInput, return_tensors: Optional[str] = "pt", **kwargs) -> BatchFeature:
        cpu_patchify = kwargs.pop("cpu_patchify", getattr(self, "cpu_patchify", False))
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError("Invalid image type.")
        image_tensors = []
        patch_sequences = []
        image_grid_thw = []
        for image in images:
            image = convert_to_rgb(image)
            resized = self.resize_image(image)
            tensor = self._to_tensor(resized)
            _, h, w = tensor.shape
            image_grid_thw.append([1, h // self.effective_patch_size, w // self.effective_patch_size])
            if cpu_patchify:
                patch_sequences.append(self._to_patch_sequence(tensor))
            else:
                image_tensors.append(tensor)
        if cpu_patchify:
            pixel_values = torch.cat(patch_sequences, dim=0)
        else:
            pixel_values = self._pad_image_tensors(image_tensors)
        return BatchFeature(data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors)


class HainaOCRNativePixelProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    VISION_START_ID = 151652
    VISION_END_ID = 151653
    IMAGE_PAD_ID = 151655

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if image_processor is None:
            image_processor = HainaOCRNativePixelImageProcessor()
        if tokenizer is None:
            raise ValueError("Tokenizer is required.")
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        if not hasattr(tokenizer, "image_token"):
            tokenizer.image_token = "<|image_pad|>"
        if not hasattr(tokenizer, "image_token_id"):
            tokenizer.image_token_id = self.IMAGE_PAD_ID
        if not hasattr(tokenizer, "vision_start_token"):
            tokenizer.vision_start_token = "<|vision_start|>"
        if not hasattr(tokenizer, "vision_end_token"):
            tokenizer.vision_end_token = "<|vision_end|>"
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(self, text: Union[str, List[str]] = None, images: ImageInput = None, return_tensors: Optional[str] = "pt", **kwargs) -> BatchFeature:
        cpu_patchify = kwargs.pop("cpu_patchify", False)
        if images is not None:
            image_inputs = self.image_processor(images, return_tensors=return_tensors, cpu_patchify=cpu_patchify)
            image_grid_thw = image_inputs.get("image_grid_thw", None)
        else:
            image_inputs = {}
            image_grid_thw = None
        if text is not None:
            truncation = kwargs.pop("truncation", True)
            padding = kwargs.pop("padding", True)
            if not isinstance(text, list):
                text = [text]
            text = text.copy()
            if image_grid_thw is not None:
                index = 0
                for i in range(len(text)):
                    text[i], index = self._expand_image_placeholders(text[i], image_grid_thw, index)
                if index != len(image_grid_thw):
                    raise ValueError("Number of images does not match placeholders found in text.")
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                **kwargs,
            )
        else:
            text_inputs = {}
        return BatchFeature(data={**image_inputs, **text_inputs}, tensor_type=return_tensors)

    def _expand_image_placeholders(self, text: str, image_grid_thw, start_index: int = 0):
        image_token = self.tokenizer.image_token
        temp_placeholder = "<|placeholder|>"
        index = start_index
        while image_token in text or "<image>" in text:
            if index >= len(image_grid_thw):
                raise ValueError("More image placeholders in text than processed images.")
            grid_t, grid_h, grid_w = image_grid_thw[index]
            num_image_tokens = int(grid_t) * int(grid_h) * int(grid_w)
            placeholder = temp_placeholder * num_image_tokens
            if image_token in text:
                text = text.replace(image_token, placeholder, 1)
            else:
                text = text.replace("<image>", placeholder, 1)
            index += 1
        return text.replace(temp_placeholder, image_token), index

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        names = self.tokenizer.model_input_names + self.image_processor.model_input_names
        return list(dict.fromkeys(names))

    def save_pretrained(self, save_directory):
        import json
        import os
        os.makedirs(save_directory, exist_ok=True)
        self.image_processor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        with open(os.path.join(save_directory, "processor_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image_processor_type": "HainaOCRNativePixelImageProcessor",
                    "tokenizer_class": "AutoTokenizer",
                    "processor_class": "HainaOCRNativePixelProcessor",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = load_tokenizer_with_fixes(pretrained_model_name_or_path, **kwargs)
        try:
            image_processor = HainaOCRNativePixelImageProcessor.from_pretrained(pretrained_model_name_or_path)
        except Exception:
            image_processor = HainaOCRNativePixelImageProcessor()
        return cls(image_processor=image_processor, tokenizer=tokenizer)

