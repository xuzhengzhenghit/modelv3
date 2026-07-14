#!/usr/bin/env python3
"""Character-level OCR tokenizer from ppocr_vocab/vocab.txt.

Compatible with HuggingFace PreTrainedTokenizer API.
Usage:
    tok = OCRTokenizer.from_pretrained('/path/to/vocab/dir')
    ids = tok.encode('Hello world', add_special_tokens=False)
    text = tok.decode(ids, skip_special_tokens=True)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizer


VOCAB_FILENAME = "vocab.txt"
CONFIG_FILENAME = "tokenizer_config.json"

# Default special token IDs (matches data/ocr_vocab/vocab.txt order)
DEFAULT_SPECIALS = {
    "pad_token": "<|pad|>",
    "eos_token": "<|eos|>",
    "unk_token": "<|unk|>",
    "bos_token": "<|im_start|>",
    "additional_special_tokens": [
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|ocr|>",
    ],
}


class OCRTokenizer(PreTrainedTokenizer):
    """Character-level tokenizer backed by a flat vocab file (one token per line)."""

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": VOCAB_FILENAME}

    def __init__(self, vocab_file: str, **kwargs):
        # Load vocab FIRST (super().__init__ calls get_vocab)
        self._vocab: dict[str, int] = {}
        self._ids_to_tokens: dict[int, str] = {}

        with open(vocab_file, "r", encoding="utf-8", newline="") as f:
            raw = f.read()
        # Split on newlines; a completely empty line = literal newline char token
        lines = raw.split("\n")
        idx = 0
        for line in lines:
            if line == "":
                token = "\n"
            else:
                token = line
            if token not in self._vocab:
                self._vocab[token] = idx
                self._ids_to_tokens[idx] = token
                idx += 1

        super().__init__(
            pad_token=kwargs.pop("pad_token", "<|pad|>"),
            eos_token=kwargs.pop("eos_token", "<|eos|>"),
            unk_token=kwargs.pop("unk_token", "<|unk|>"),
            bos_token=kwargs.pop("bos_token", "<|im_start|>"),
            **kwargs,
        )

        self._unk_token_id = self._vocab.get(str(self.unk_token), 2)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        """Character-level tokenization — each char is a token."""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, str(self.unk_token))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        path = os.path.join(save_directory, (filename_prefix or "") + VOCAB_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            for idx in range(len(self._ids_to_tokens)):
                f.write(self._ids_to_tokens[idx] + "\n")
        return (path,)

    @classmethod
    def from_vocab_dir(cls, vocab_dir: str | Path, **kwargs) -> "OCRTokenizer":
        """Load from a directory containing vocab.txt."""
        vocab_dir = Path(vocab_dir)
        vocab_file = vocab_dir / VOCAB_FILENAME
        if not vocab_file.exists():
            raise FileNotFoundError(f"vocab.txt not found in {vocab_dir}")
        return cls(str(vocab_file), **kwargs)


def build_and_save_tokenizer(vocab_path: str, save_dir: str) -> OCRTokenizer:
    """Build tokenizer from vocab.txt and save to directory."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tok = OCRTokenizer(vocab_path)
    tok.save_pretrained(str(save_dir))

    print(f"Tokenizer saved to {save_dir}")
    print(f"  vocab_size: {tok.vocab_size}")
    print(f"  pad_token: {tok.pad_token} (id={tok.pad_token_id})")
    print(f"  eos_token: {tok.eos_token} (id={tok.eos_token_id})")
    print(f"  unk_token: {tok.unk_token} (id={tok.unk_token_id})")
    return tok


if __name__ == "__main__":
    import sys
    vocab_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/si001719bp3c/default/XJZ/modelv3/data/ocr_vocab/vocab.txt"
    save_dir = sys.argv[2] if len(sys.argv) > 2 else "/mnt/si001719bp3c/default/XJZ/modelv3/data/ocr_vocab/tokenizer"
    build_and_save_tokenizer(vocab_path, save_dir)
