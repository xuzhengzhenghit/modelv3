#!/usr/bin/env python3
"""Test: formula normalization roundtrip — parse → canonical form is consistent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))

from document_parser import canonical_display, parse_inline_parts, parse_document


def test_canonical_display_bracket():
    tex, target = canonical_display(r"\[x^2 + y^2 = z^2\]")
    assert "x^2" in tex
    assert target.startswith(r"\[")
    assert target.endswith(r"\]")


def test_canonical_display_double_dollar():
    tex, target = canonical_display(r"$$E = mc^2$$")
    assert "E = mc^2" in tex
    assert target.startswith(r"\[")


def test_canonical_display_latex_tag():
    tex, target = canonical_display(r"[latex]\alpha + \beta[/latex]")
    assert r"\alpha" in tex
    assert target.startswith(r"\[")


def test_inline_math_braces():
    parts = parse_inline_parts(r"Hello \(x=1\) world")
    assert len(parts) == 3
    assert parts[1]["kind"] == "math_inline"
    assert parts[1]["target"] == r"\(x=1\)"


def test_parse_document_mixed():
    text = "Text with $$E=mc^2$$ and \\(x=1\\) inline."
    blocks = parse_document(text, "Physics")
    assert len(blocks) >= 2
    assert any(b["kind"] == "math_display" for b in blocks)
    assert any(
        any(p.get("kind") == "math_inline" for p in b["parts"])
        for b in blocks
        if b["kind"] == "paragraph"
    )
