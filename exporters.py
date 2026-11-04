# ==============================================================================
# MODULE: exporters.py
# Version: v2.9.6 | Date: 2026-11-04
# Purpose: HTML feed rendering, Unicode-to-Latin1 PDF sanitizer, paginated PDF
#          transcript generator, and structured JSON trace exporter.
# ==============================================================================

import os
import re
import json
import tempfile
import unicodedata
from datetime import datetime
import markdown
import gradio as gr
from xhtml2pdf import pisa

from config import (
    APP_VERSION,
    TRACE,
    DEFAULT_MAX_LOOPS,
    IST_TZ
)

def render_feed_html(topic: str, history: list, max_loops: int) -> str:
    """Renders the scrollable stack of Supporter and Contender cards."""
    if not history:
        return "<div style='padding:20px; text-align:center; color:#6b7280;'>Enter any proposition above and click <b>Start Debate</b> to begin Round 1.</div>"

    html_blocks = [f"<h3 style='margin-bottom:14px;'>Proposition: <em>{topic}</em></h3>"]
    for msg in history:
        body_html = markdown.markdown(msg["content"])
        if msg["role"] == "Supporter":
            html_blocks.append(f"""
            <div class="supporter-card">
                <div class="supporter-header">
                    <span>🟢 Agent A: The Supporter • Groq ({msg['model']})</span>
                    <span class="round-badge">Round {msg['round']} of {max_loops}</span>
                </div>
                <div class="card-body">{body_html}</div>
            </div>
            """)
        else:
            html_blocks.append(f"""
            <div class="contender-card">
                <div class="contender-header">
                    <span>🔴 Agent B: The Contender • Gemini ({msg['model']})</span>
                    <span class="round-badge">Round {msg['round']} of {max_loops}</span>
                </div>
                <div class="card-body">{body_html}</div>
            </div>
            """)

    return "\n".join(html_blocks)

def sanitize_for_pdf(text: str) -> str:
    """Normalizes subscripts/superscripts (e.g. CO₂ -> CO2) and converts Unicode punctuation for xhtml2pdf."""
    if not text:
        return ""

    replacements = {
        "\u2010": "-",   # Hyphen
        "\u2011": "-",   # Non-breaking hyphen
        "\u2012": "-",   # Figure dash
        "\u2013": "-",   # En dash
        "\u2014": "--",  # Em dash
        "\u2015": "--",  # Horizontal bar
        "\u2212": "-",   # Mathematical minus sign
        "\u2043": "-",   # Hyphen bullet
        "\ufe58": "-",   # Small em dash
        "\ufe63": "-",   # Small hyphen-minus
        "\uff0d": "-",   # Fullwidth hyphen-minus
        "\u2018": "'",   # Left single smart quote
        "\u2019": "'",   # Right single smart quote / apostrophe
        "\u201a": "'",   # Single low-9 quote
        "\u201b": "'",   # Single high-reversed-9 quote
        "\u2032": "'",   # Prime
        "\u201c": '"',   # Left double smart quote
        "\u201d": '"',   # Right double smart quote
        "\u201e": '"',   # Double low-9 quote
        "\u2033": '"',   # Double prime
        "\u00ab": '"',   # Left guillemet
        "\u00bb": '"',   # Right guillemet
        "\u2026": "...", # Horizontal ellipsis
        "\u00a0": " ",   # Non-breaking space
        "\u202f": " ",   # Narrow no-break space
        "\u2009": " ",   # Thin space
        "\u200b": "",    # Zero-width space
        "\u2022": "*",   # Bullet
        "\u2264": "<=",  # Less-than or equal to
        "\u2265": ">=",  # Greater-than or equal to
        "\u2260": "!=",  # Not equal to
        "\u00d7": "x",   # Multiplication sign
        "\u20b9": "Rs.", # Indian Rupee symbol
    }

    for unicode_char, ascii_equiv in replacements.items():
        text = text.replace(unicode_char, ascii_equiv)

    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def _get_clean_topic_slug(raw_topic: str) -> str:
    """Helper to generate a clean filename slug from the debate topic."""
    safe_topic = sanitize_for_pdf(raw_topic or "Debate")
    clean_topic = re.sub(r'[^a-zA-Z0-9_\- ]', '', safe_topic).strip().replace(" ", "_")[:40]
    return clean_topic if clean_topic else "Debate"

def export_transcript_file(state):
    """Generates a styled, paginated PDF file containing the full debate history."""
    if not state or not state.get("history"):
        raise gr.Error("No debate content to export yet! Run at least one turn first.")

    raw_topic = state.get("topic", "Debate")
    safe_topic = sanitize_for_pdf(raw_topic)
    clean_slug = _get_clean_topic_slug(raw_topic)
    filename = f"Debate_{clean_slug}.pdf"

    total_turns_done = state.get("current_turn", 0)
    rounds_completed = (total_turns_done + 1) // 2
    now_ist = datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S IST')

    pdf_html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: letter;
            margin: 1.5cm;
            @bottom-right {{ content: counter(page); }}
        }}
        body {{
            font-family: Helvetica, Arial, sans-serif;
            color: #1f2937;
            font-size: 10pt;
            line-height: 1.48;
        }}
        h1 {{
            font-size: 16pt;
            color: #111827;
            margin-bottom: 4px;
        }}
        .meta {{
            font-size: 9pt;
            color: #6b7280;
            margin-bottom: 15px;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 8px;
        }}
        .supporter-card {{
            background-color: #f0fdf4;
            border-left: 4px solid #16a34a;
            padding: 10px 14px;
            margin-bottom: 12px;
        }}
        .contender-card {{
            background-color: #fef2f2;
            border-left: 4px solid #dc2626;
            padding: 10px 14px;
            margin-bottom: 12px;
        }}
        .header-sup {{
            font-weight: bold;
            color: #15803d;
            font-size: 10pt;
            margin-bottom: 6px;
        }}
        .header-con {{
            font-weight: bold;
            color: #b91c1c;
            font-size: 10pt;
            margin-bottom: 6px;
        }}
        p, li {{
            margin-bottom: 4px;
        }}
    </style>
    </head>
    <body>
        <h1>Multi-Agent AI Debate Transcript ({APP_VERSION})</h1>
        <div class="meta">
            <b>Proposition:</b> {safe_topic}<br/>
            <b>Date:</b> {now_ist} | 
            <b>Progress:</b> {total_turns_done} turns ({rounds_completed} of {state.get('max_loops', DEFAULT_MAX_LOOPS)} rounds)
        </div>
    """

    for item in state["history"]:
        sanitized_markdown = sanitize_for_pdf(item["content"])
        body_html = markdown.markdown(sanitized_markdown)
        safe_model = sanitize_for_pdf(str(item.get("model", "")))
        if item["role"] == "Supporter":
            pdf_html += f"""
            <div class="supporter-card">
                <div class="header-sup">Agent A: The Supporter - Groq ({safe_model}) - Round {item['round']}</div>
                {body_html}
            </div>
            """
        else:
            pdf_html += f"""
            <div class="contender-card">
                <div class="header-con">Agent B: The Contender - Gemini ({safe_model}) - Round {item['round']}</div>
                {body_html}
            </div>
            """

    pdf_html += "</body></html>"

    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    
    with open(filepath, "w+b") as f:
        pisa_status = pisa.CreatePDF(pdf_html, dest=f)
        
    if pisa_status.err:
        raise gr.Error("PDF generation error occurred.")

    return filepath

def export_trace_json_file(state):
    """Generates a structured JSON file '{topic}_trace_log.json' when TRACE is ON."""
    if not state or not state.get("trace_logs"):
        return None

    raw_topic = state.get("topic", "Debate")
    clean_slug = _get_clean_topic_slug(raw_topic).lower()
    filename = f"{clean_slug}_trace_log.json"

    payload = {
        "engine_version": APP_VERSION,
        "trace_enabled": TRACE,
        "topic": raw_topic,
        "exported_at_ist": datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S IST'),
        "max_rounds_configured": state.get("max_loops", DEFAULT_MAX_LOOPS),
        "total_turns_recorded": len(state.get("trace_logs", [])),
        "turns": state.get("trace_logs", [])
    }

    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return filepath