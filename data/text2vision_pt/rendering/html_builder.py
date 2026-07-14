#!/usr/bin/env python3
"""HTML builder — construct safe HTML/CSS pages for scientific content blocks."""

from __future__ import annotations

import re
from typing import Any

from .style_sampler import RenderSpec


# HTML template with KaTeX support
PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{
    margin: 0; padding: 0;
    width: {width}px; height: {height}px;
    overflow: hidden;
    background: {bg_color};
}}
body {{
    {body_css}
}}
.content {{
    max-width: {max_width}px;
    word-wrap: break-word;
    overflow-wrap: break-word;
}}
.katex-display {{ margin: 0.5em 0 !important; }}
.katex {{ font-size: {katex_size}em !important; }}
table {{
    border-collapse: collapse;
    width: 100%;
    max-width: 100%;
    margin: 0.5em 0;
    font-size: {table_font_size}%;
}}
table td, table th {{
    border: {table_border};
    padding: 4px 8px;
    text-align: left;
}}
table th {{
    background: #f0f0f0;
    font-weight: bold;
}}
.math-display-wrapper {{
    margin: 0.5em 0;
    text-align: center;
}}
.math-inline {{
    white-space: nowrap;
}}
</style>
<link rel="stylesheet" href="{katex_css}">
</head>
<body>
<div class="content">
{body_html}
</div>
<script src="{katex_js}"></script>
<script>
(function() {{
    try {{
        var els = document.querySelectorAll('.math-render');
        for (var i = 0; i < els.length; i++) {{
            var el = els[i];
            var tex = el.getAttribute('data-tex') || el.textContent || '';
            var display = el.classList.contains('math-display');
            try {{
                katex.render(tex, el, {{ displayMode: display, throwOnError: true }});
            }} catch(e) {{
                el.textContent = tex;
                el.classList.add('katex-error');
            }}
        }}
    }} catch(e) {{}}
}})();
</script>
</body>
</html>
"""

# Minimal measurement template (no fixed dimensions, used for content sizing)
MEASURE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{
    margin: 0; padding: 0;
    background: {bg_color};
}}
body {{
    {body_css}
}}
.content {{
    max-width: {max_width}px;
    word-wrap: break-word;
    overflow-wrap: break-word;
    display: inline-block;
}}
.katex-display {{ margin: 0.5em 0 !important; }}
.katex {{ font-size: {katex_size}em !important; }}
table td, table th {{
    border: {table_border};
    padding: 4px 8px;
    text-align: left;
}}
</style>
<link rel="stylesheet" href="{katex_css}">
</head>
<body>
<div class="content" id="measure-target">
{body_html}
</div>
<script src="{katex_js}"></script>
<script>
(function() {{
    try {{
        var els = document.querySelectorAll('.math-render');
        for (var i = 0; i < els.length; i++) {{
            var el = els[i];
            var tex = el.getAttribute('data-tex') || el.textContent || '';
            var display = el.classList.contains('math-display');
            try {{
                katex.render(tex, el, {{ displayMode: display, throwOnError: true }});
            }} catch(e) {{
                el.textContent = tex;
                el.classList.add('katex-error');
            }}
        }}
    }} catch(e) {{}}
}})();
</script>
</body>
</html>
"""

MEASUREMENT_JS = """
(function() {
    var el = document.getElementById('measure-target');
    if (!el) return JSON.stringify({width: 0, height: 0, error: 'no-element'});
    var rect = el.getBoundingClientRect();
    return JSON.stringify({
        width: Math.ceil(rect.width),
        height: Math.ceil(rect.height),
    });
})();
"""


def _escape_html(text: str) -> str:
    """Escape text for safe HTML insertion."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _render_block_as_html(block: dict[str, Any]) -> str:
    """Convert a single block to HTML."""
    kind = block.get("kind")
    if kind == "paragraph":
        parts = []
        for part in block.get("parts", []):
            if part.get("kind") == "math_inline":
                tex = part.get("tex", "")
                parts.append(
                    f'<span class="math-render math-inline" data-tex="{_escape_html(tex)}">{_escape_html(tex)}</span>'
                )
            else:
                parts.append(_escape_html(part.get("text", "")))
        return f"<p>{''.join(parts)}</p>"

    elif kind == "math_display":
        tex = block.get("tex", "")
        return (
            f'<div class="math-display-wrapper">'
            f'<span class="math-render math-display" data-tex="{_escape_html(tex)}">{_escape_html(tex)}</span>'
            f"</div>"
        )

    elif kind == "table":
        rows = block.get("rows", [])
        if not rows:
            return ""
        html_parts = ["<table><tbody>"]
        for row in rows:
            html_parts.append("<tr>")
            for cell in row:
                html_parts.append(f"<td>{_escape_html(str(cell))}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody></table>")
        return "\n".join(html_parts)

    return ""


def build_page_html(
    blocks: list[dict[str, Any]],
    spec: RenderSpec,
    canvas_width: int,
    canvas_height: int,
    katex_css: str,
    katex_js: str,
) -> str:
    """Build a complete HTML page for screenshot rendering."""
    body_parts = []
    for block in blocks:
        html_block = _render_block_as_html(block)
        if html_block:
            body_parts.append(html_block)

    body_html = "\n".join(body_parts)

    return PAGE_TEMPLATE.format(
        width=canvas_width,
        height=canvas_height,
        bg_color=spec.bg_color,
        body_css=_css_for_spec(spec, is_measure=False),
        max_width=spec.max_content_width,
        katex_size=spec.formula_size_mult,
        table_font_size=spec.table_font_size_pct,
        table_border=spec.table_border,
        katex_css=katex_css,
        katex_js=katex_js,
        body_html=body_html,
    )


def build_measure_html(
    blocks: list[dict[str, Any]],
    spec: RenderSpec,
    katex_css: str,
    katex_js: str,
) -> str:
    """Build an HTML page for content size measurement."""
    body_parts = []
    for block in blocks:
        html_block = _render_block_as_html(block)
        if html_block:
            body_parts.append(html_block)

    body_html = "\n".join(body_parts)

    return MEASURE_TEMPLATE.format(
        bg_color=spec.bg_color,
        body_css=_css_for_spec(spec, is_measure=True),
        max_width=spec.max_content_width,
        katex_size=spec.formula_size_mult,
        table_border=spec.table_border,
        katex_css=katex_css,
        katex_js=katex_js,
        body_html=body_html,
    )


def _css_for_spec(spec: RenderSpec, is_measure: bool = False) -> str:
    """Generate CSS for a RenderSpec."""
    padding = ""
    if not is_measure:
        padding = f"padding: {spec.top_margin}px {spec.right_margin}px {spec.bottom_margin}px {spec.left_margin}px;"

    return f"""
    font-family: {spec.font_family};
    font-size: {spec.font_size}px;
    line-height: {spec.line_height};
    letter-spacing: {spec.letter_spacing}em;
    text-align: {spec.text_align};
    font-weight: {spec.font_weight};
    font-style: {spec.font_style};
    color: {spec.text_color};
    {padding}
"""
