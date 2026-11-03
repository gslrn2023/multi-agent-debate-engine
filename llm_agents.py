# ==============================================================================
# MODULE: llm_agents.py
# Version: v2.9.2 | Date: 2026-11-03
# Purpose: Prompt loading, rate-limit backoff, isolated Groq (Supporter) and
#          Gemini (Contender) API execution, role-tagged transcript builder,
#          and output markdown sanitization.
# ==============================================================================

import os
import re
import time
import gradio as gr
from groq import Groq
from google import genai
from google.genai import types

from config import (
    APP_VERSION,
    SUPPORTER_PROMPT_FILE,
    CONTENDER_PROMPT_FILE
)

def load_system_prompt(agent_role: str, topic: str, current_round: int, max_rounds: int, is_very_first_speech: bool) -> str:
    """Loads the prompt from supporter_prompt.txt or contender_prompt.txt and injects round variables."""
    file_path = SUPPORTER_PROMPT_FILE if agent_role == "Supporter" else CONTENDER_PROMPT_FILE

    if not os.path.exists(file_path):
        raise gr.Error(f"Missing prompt file: '{file_path}'. Please create it in your Space Files tab.")

    with open(file_path, "r", encoding="utf-8") as f:
        template = f.read()

    if is_very_first_speech:
        phase_goal = (
            "Opening Case (Round 1) — Deliver 4 to 5 hard-hitting, evidence-dense bullets (30-40 words each) "
            "establishing your core case with verified statistics, empirical findings, and named sources."
        )
    elif current_round < max_rounds:
        phase_goal = (
            f"Rebuttal & Expansion (Round {current_round}) — Deliver 4 to 5 sharp bullets (30-40 words each): "
            "use your first 2 to 3 bullets to directly dismantle the opponent's specific claims from the last turn, and "
            "your final 1 to 2 bullets to introduce new empirical evidence from this round's research."
        )
    else:
        phase_goal = (
            f"Final Closing Clash (Round {current_round}) — Deliver 4 to 5 decisive bullets (30-40 words each): "
            "use your first 2 to 3 bullets to refute the opponent's last turn, and your final 1 to 2 bullets "
            "to lock in the overarching comparative evidence."
        )

    populated = (
        template
        .replace("{topic}", topic)
        .replace("{current_round}", str(current_round))
        .replace("{max_rounds}", str(max_rounds))
        .replace("{phase_goal}", phase_goal)
    )

    return f"{populated}\n\n--- LIVE DEBATE CONTEXT ({APP_VERSION}) ---\nProposition: \"{topic}\"\nCurrent Round: {current_round} of {max_rounds}\nRound Objective: {phase_goal}"

def enforce_wait_time(last_call_ts: float, wait_sec: int, next_agent_label: str):
    """Strictly pauses execution between EACH AI agent turn if clicked faster than wait_sec."""
    if last_call_ts > 0 and wait_sec > 0:
        elapsed = time.time() - last_call_ts
        remaining = wait_sec - elapsed
        if remaining > 0:
            gr.Info(f"⏳ Pacing API limits before {next_agent_label}: waiting {remaining:.1f}s...")
            time.sleep(remaining)

def call_with_rate_limit_retry(api_func, provider_name: str, max_retries: int = 3, base_wait: int = 15, raise_ui_error: bool = True) -> str:
    """Executes an API call and automatically backs off on 429 / TPM / Quota errors."""
    for attempt in range(max_retries):
        try:
            return api_func()
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = any(k in err_msg for k in ["429", "rate_limit", "rate limit", "quota", "resource_exhausted", "tpm"])
            if is_rate_limit and attempt < max_retries - 1:
                wait_seconds = base_wait * (attempt + 1)
                gr.Warning(f"⚠️ {provider_name} rate limit reached. Auto-waiting {wait_seconds}s before retry ({attempt + 1}/{max_retries - 1})...")
                time.sleep(wait_seconds)
            else:
                if raise_ui_error:
                    raise gr.Error(f"{provider_name} API Error: {str(e)}")
                raise e

def call_groq_supporter(system_prompt: str, user_prompt: str, groq_model: str, temperature: float = 0.35) -> str:
    """Calls Groq API exclusively for the Supporter agent."""
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise gr.Error("Missing GROQ_API_KEY in Space Secrets.")

    def _execute():
        client = Groq(api_key=groq_key)
        combined_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        kwargs = {
            "model": groq_model,
            "messages": [{"role": "user", "content": combined_prompt}],
            "temperature": temperature,
            "max_completion_tokens": 1800,
        }
        if "gpt-oss" in groq_model:
            kwargs["reasoning_effort"] = "low"

        completion = client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty response. Try clicking Next Turn again.")
        return content

    return call_with_rate_limit_retry(_execute, "Groq (Supporter)")

def call_gemini_contender(system_prompt: str, user_prompt: str, gemini_model: str, temperature: float = 0.35) -> str:
    """Calls Gemini API exclusively for the Contender agent."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        raise gr.Error("Missing GEMINI_API_KEY in Space Secrets.")

    def _execute():
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model=gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=3072,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=1024
                ),
            )
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response. Try clicking Next Turn again.")
        return response.text

    return call_with_rate_limit_retry(_execute, "Gemini (Contender)")

def build_tpm_safe_transcript(history: list, current_agent_role: str) -> str:
    """
    Formats the debate history with explicit ownership tags ('YOUR PREVIOUS SPEECH' vs.
    'OPPONENT'S SPEECH TO REFUTE') so agents never accidentally refute their own prior points.
    """
    if len(history) <= 6:
        selected = history
    else:
        selected = (
            history[:2]
            + [{"role": "System", "round": "...", "content": "*[Earlier middle rounds omitted to preserve token budget]*"}]
            + history[-4:]
        )

    blocks = []
    for m in selected:
        if m["role"] == "System":
            blocks.append(m["content"])
        elif m["role"] == current_agent_role:
            blocks.append(
                f"[Round {m['round']} — YOUR PREVIOUS SPEECH ({m['role']}) — Maintain consistency; do NOT repeat or contradict these points]:\n{m['content']}"
            )
        else:
            blocks.append(
                f"[Round {m['round']} — OPPONENT'S SPEECH TO REFUTE ({m['role']}) — Target and dismantle these claims]:\n{m['content']}"
            )
    return "\n\n".join(blocks)

def clean_agent_output(text: str) -> str:
    """
    1. Strips standalone section headers ('Opening Hook', 'Fact-Sheet', 'Closing Takeaway').
    2. Strips mechanical bullet prefixes ('Direct Rebuttal:', 'Methodological audit -').
    3. Ensures a blank line before and after the bullet list so markdown renders proper HTML <ul><li> lists.
    """
    if not text:
        return ""

    standalone_section_pattern = re.compile(
        r'^\s*(?:\#+\s*|\*\*\s*)?'
        r'(?:Opening\s+Hook|Opening\s+Paragraph|Fact[\-\s\u2011]Sheet(?:\s+Core)?|Point[\-\s\u2011]Wise\s+Core|Closing\s+Takeaway|Closing\s+Paragraph)'
        r'(?:\s*\*\*)?\s*[:\-–—]?\s*$',
        re.IGNORECASE
    )

    inline_section_prefix = re.compile(
        r'^\s*(\*\*)?(?:Opening\s+Hook|Fact[\-\s\u2011]Sheet(?:\s+Core)?|Closing\s+Takeaway)(\*\*)?\s*[:\-–—]\s*',
        re.IGNORECASE
    )

    outer_bullet_pattern = re.compile(
        r'^(\s*[\*\-\•]\s*)(\*\*)?'
        r'(?:'
        r'Direct\s+Rebuttal|Rebuttal(?:\s+to\s+[^:–\-]{1,35})?|'
        r'New\s+Independent\s+Front|New\s+Front|Independent\s+Front|'
        r'Mode\s+[ABC](?:\s*\([^\)]{1,35}\))?|'
        r'(?:Empirical|Methodological|Baseline|Structural|Causal|Scope|Premise)(?:\s*/\s*(?:Baseline|Causal|Scope|Structural))?\s+(?:Counter[\-\s]?)?Audit'
        r')'
        r'(\*\*)?\s*[:\-–—]\s*(\*\*)?',
        re.IGNORECASE
    )

    inline_tag_pattern = re.compile(r'\b(?:Deep\s+Study|Publication|Source|Web\s+Synthesis)\s*:\s*', re.IGNORECASE)

    cleaned_lines = []
    for raw_line in text.splitlines():
        if standalone_section_pattern.match(raw_line):
            continue

        line = inline_section_prefix.sub("", raw_line)

        def _replacer(match):
            trailing_bold = match.group(4) or match.group(2) or ""
            return f"* {trailing_bold}"

        line = outer_bullet_pattern.sub(_replacer, line)
        line = re.sub(r'^\s*[\*\-\•]\s+', '* ', line)
        line = inline_tag_pattern.sub("", line)
        cleaned_lines.append(line)

    final_lines = []
    for line in cleaned_lines:
        is_bullet = line.startswith("* ")
        prev_line = final_lines[-1] if final_lines else ""
        if is_bullet and prev_line.strip() != "" and not prev_line.startswith("* "):
            final_lines.append("")
        if not is_bullet and line.strip() != "" and prev_line.startswith("* "):
            final_lines.append("")
        final_lines.append(line)

    return "\n".join(final_lines).strip()