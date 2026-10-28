import os
import re
import json
import time
import tempfile
import threading
import unicodedata
from datetime import datetime, timedelta, timezone
import markdown
import gradio as gr
import spaces
from groq import Groq
from google import genai
from google.genai import types
from tavily import TavilyClient
from exa_py import Exa
from xhtml2pdf import pisa

# ==========================================
# CONFIGURABLE VARIABLES (LOOPS, WAIT & DAILY LIMIT)
# ==========================================
DEFAULT_MAX_LOOPS = 3       # // adjust the value here // (Default & Max number of debate loops/rounds)
DEFAULT_WAIT_TIME_SEC = 30  # // adjust the value here // (Wait time in seconds between EACH AI agent execution)
MAX_DAILY_DEBATES = 5       # // adjust the value here // (Global max debates per day across all users)
DAILY_LIMIT_FILE = "daily_limit.json"
DAILY_LIMIT_LOCK = threading.Lock()
IST_TZ = timezone(timedelta(hours=5, minutes=30))

# ==========================================
# PROMPT FILE NAMES (TWO SEPARATE FILES)
# ==========================================
SUPPORTER_PROMPT_FILE = "supporter_prompt.txt"
CONTENDER_PROMPT_FILE = "contender_prompt.txt"

# ==========================================
# ZEROGPU STARTUP REQUIREMENT HANDLER
# ==========================================
@spaces.GPU(duration=5)
def zerogpu_healthcheck():
    """Satisfies Hugging Face ZeroGPU mandatory @spaces.GPU startup detection."""
    return True

# ==========================================
# CUSTOM CSS FOR DISTINCT AGENT & JUDGE PALETTES
# ==========================================
CUSTOM_CSS = """
.supporter-card {
    background-color: #f0fdf4;
    border-left: 6px solid #16a34a;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 18px;
    color: #064e3b;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.supporter-header {
    font-weight: 700;
    font-size: 1.08rem;
    color: #15803d;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #bbf7d0;
    padding-bottom: 6px;
}
.contender-card {
    background-color: #fef2f2;
    border-left: 6px solid #dc2626;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 18px;
    color: #7f1d1d;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.contender-header {
    font-weight: 700;
    font-size: 1.08rem;
    color: #b91c1c;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #fecaca;
    padding-bottom: 6px;
}
.judge-card {
    background-color: #fefce8;
    border-left: 6px solid #ca8a04;
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 22px;
    margin-bottom: 18px;
    color: #422006;
    box-shadow: 0 3px 8px rgba(0,0,0,0.08);
}
.judge-header {
    font-weight: 700;
    font-size: 1.12rem;
    color: #a16207;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #fde047;
    padding-bottom: 6px;
}
.round-badge {
    font-size: 0.85rem;
    padding: 2px 10px;
    border-radius: 12px;
    background-color: rgba(0,0,0,0.08);
}
.card-body p, .card-body li {
    color: inherit !important;
    line-height: 1.6;
}
"""

# ==========================================
# THREAD-SAFE DAILY DEBATE LIMITER (IST)
# ==========================================
def _read_daily_usage_unlocked(today_str: str) -> dict:
    """Internal helper to read and auto-reset daily usage data (must be called inside DAILY_LIMIT_LOCK)."""
    if os.path.exists(DAILY_LIMIT_FILE):
        try:
            with open(DAILY_LIMIT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"date": today_str, "count": 0}
    else:
        data = {"date": today_str, "count": 0}

    if data.get("date") != today_str:
        data = {"date": today_str, "count": 0}

    return data

def can_start_new_debate() -> tuple[bool, int]:
    """Thread-safe check to verify if daily debate quota is available before starting Turn 1."""
    today_str = datetime.now(IST_TZ).strftime("%Y-%m-%d")
    with DAILY_LIMIT_LOCK:
        data = _read_daily_usage_unlocked(today_str)
        current_count = int(data.get("count", 0))
        return (current_count < MAX_DAILY_DEBATES), current_count

def record_new_debate_started() -> int:
    """Thread-safe increment called ONLY after Turn 1 succeeds so failed API calls don't burn quota."""
    today_str = datetime.now(IST_TZ).strftime("%Y-%m-%d")
    with DAILY_LIMIT_LOCK:
        data = _read_daily_usage_unlocked(today_str)
        data["count"] = int(data.get("count", 0)) + 1
        with open(DAILY_LIMIT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data["count"]

# ==========================================
# PASSWORD VERIFICATION (CHK_PSSWD)
# ==========================================
def verify_password(entered_password: str):
    """Checks user password against the CHK_PSSWD secret in Hugging Face Settings."""
    expected_password = os.environ.get("CHK_PSSWD", "").strip()
    if not expected_password:
        raise gr.Error("Missing 'CHK_PSSWD' secret! Please set CHK_PSSWD in your Hugging Face Space Settings -> Variables and secrets.")

    if entered_password.strip() == expected_password:
        return (
            gr.update(visible=False),  # Hide login container
            gr.update(visible=True),   # Show main debate application
            ""                         # Clear any error message
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "❌ **Incorrect password.** Please try again."
        )

# ==========================================
# DYNAMIC SEARCH AGGREGATION (TAVILY + EXA)
# ==========================================
def extract_opponent_focus_terms(last_turn_text: str) -> str:
    """Extracts key terms from the opponent's last speech so Round 2+ searches pull fresh counter-evidence."""
    if not last_turn_text:
        return ""
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "while", "are", "was", "were",
        "have", "has", "had", "not", "but", "what", "all", "can", "more", "than", "into",
        "over", "under", "their", "they", "its", "also", "such", "only", "which", "when"
    }
    words = re.findall(r'\b[A-Za-z][A-Za-z\-]{4,18}\b', last_turn_text)
    seen = set()
    keywords = []
    for w in words:
        lw = w.lower()
        if lw not in stop_words and lw not in seen:
            seen.add(lw)
            keywords.append(w)
        if len(keywords) >= 6:
            break
    return " ".join(keywords)

def fetch_combined_research(topic: str, agent_role: str, current_round: int, last_opponent_text: str = "") -> str:
    """Rotates search lenses by round and injects opponent keywords so each round fetches brand-new sources."""
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    exa_key = os.environ.get("EXA_API_KEY", "").strip()

    if not tavily_key and not exa_key:
        raise gr.Error("Missing both TAVILY_API_KEY and EXA_API_KEY in Hugging Face Space Secrets.")

    opp_terms = extract_opponent_focus_terms(last_opponent_text)

    # Rotate analytical lens per round so Round 2 & Round 3 never repeat Round 1 queries
    if current_round == 1:
        sup_tavily = "macroeconomic benefits fiscal savings empirical statistics government policy report"
        sup_exa = f"Comprehensive empirical study and economic data proving the direct benefits and cost savings of {topic}"
        con_tavily = "efficiency losses hidden subsidies capital costs empirical criticisms against"
        con_exa = f"Critical policy analysis, structural trade-offs, and hidden fiscal costs challenging {topic}"
    elif current_round == 2:
        sup_tavily = f"lifecycle emissions mitigation rural employment supply chain solutions {opp_terms}"
        sup_exa = f"Scientific field trials, lifecycle optimization, and second-order economic gains refuting criticisms of {topic} ({opp_terms})"
        con_tavily = f"lifecycle environmental footprint ecological strain resource diversion supply chain bottlenecks {opp_terms}"
        con_exa = f"Deep environmental lifecycle assessment, resource depletion, and unintended systemic consequences of {topic} ({opp_terms})"
    else:
        sup_tavily = f"long-term strategic resilience global benchmarks technological scaling success {opp_terms}"
        sup_exa = f"Long-term comparative global benchmarks, technological scalability, and macro-resilience supporting {topic}"
        con_tavily = f"superior policy alternatives long-term market distortions international failure precedents {opp_terms}"
        con_exa = f"International failure cases, long-term market distortions, and superior policy alternatives to {topic}"

    if agent_role == "Supporter":
        tavily_focus, exa_focus = sup_tavily, sup_exa
    else:
        tavily_focus, exa_focus = con_tavily, con_exa

    snippets = []

    # 1. TAVILY SEARCH (Top 2 Mainstream / Statistical Results)
    if tavily_key:
        try:
            tavily_client = TavilyClient(api_key=tavily_key)
            t_query = f"{topic[:200]} {tavily_focus}"[:380]
            t_resp = tavily_client.search(
                query=t_query,
                search_depth="basic",
                max_results=2,
                include_answer=True
            )
            if t_resp.get("answer"):
                snippets.append(f"- **[Tavily Overview]:** {t_resp['answer'][:260]}")

            for res in t_resp.get("results", []):
                title = res.get("title", "Verified Report").strip()
                url = res.get("url", "")
                content = res.get("content", "").strip().replace("\n", " ")[:280]
                snippets.append(f"- **[Publication: {title} | {url}]:** {content}...")
        except Exception as e:
            gr.Warning(f"⚠️ Tavily search warning: {str(e)}. Continuing with Exa/internal data.")

    # 2. EXA NEURAL SEARCH (Top 2 Deep / Independent Front Results)
    if exa_key:
        try:
            exa_client = Exa(api_key=exa_key)
            e_resp = exa_client.search_and_contents(
                exa_focus[:360],
                type="auto",
                num_results=2,
                text={"max_characters": 320}
            )
            for res in getattr(e_resp, "results", []):
                title = (getattr(res, "title", None) or "Research Analysis").strip()
                url = getattr(res, "url", "")
                text_body = (getattr(res, "text", "") or "").strip().replace("\n", " ")[:280]
                if text_body:
                    snippets.append(f"- **[Deep Study: {title} | {url}]:** {text_body}...")
        except Exception as e:
            gr.Warning(f"⚠️ Exa search warning: {str(e)}. Continuing with Tavily/internal data.")

    if not snippets:
        return "No external web snippets returned; rely on verified historical and empirical data."

    return "\n".join(snippets)

# ==========================================
# OUTPUT SANITIZER (STRIPS LEAKED LABELS)
# ==========================================
def clean_agent_output(text: str) -> str:
    """Strips mechanical prefixes like 'Direct Rebuttal:', 'Rebuttal to X:', or 'New Independent Front:' from bullets."""
    if not text:
        return ""

    cleaned_lines = []
    label_pattern = re.compile(
        r'^(\s*[\*\-\•]\s*)(\*\*)?'
        r'(?:Direct\s+Rebuttal|Rebuttal(?:\s+to\s+[^:–\-]{1,35})?|New\s+Independent\s+Front|New\s+Front|Independent\s+Front|Opening\s+Hook|Closing\s+Takeaway)'
        r'(\*\*)?\s*[:\-–—]\s*(\*\*)?',
        re.IGNORECASE
    )

    for line in text.splitlines():
        def _replacer(match):
            bullet_prefix = match.group(1)
            trailing_bold = match.group(4) or match.group(2) or ""
            return f"{bullet_prefix}{trailing_bold}"

        cleaned_line = label_pattern.sub(_replacer, line)
        cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)

# ==========================================
# PHASE 2: EXTERNAL PROMPT LOADER & API CALLS
# ==========================================
def load_system_prompt(agent_role: str, topic: str, current_round: int, max_rounds: int, is_very_first_speech: bool) -> str:
    """Loads the prompt from supporter_prompt.txt or contender_prompt.txt and injects round variables."""
    file_path = SUPPORTER_PROMPT_FILE if agent_role == "Supporter" else CONTENDER_PROMPT_FILE

    if not os.path.exists(file_path):
        raise gr.Error(f"Missing prompt file: '{file_path}'. Please create it in your Hugging Face Space Files tab.")

    with open(file_path, "r", encoding="utf-8") as f:
        template = f.read()

    if is_very_first_speech:
        phase_goal = (
            "Opening Speech (Round 1) — Deliver 4 to 5 offensive bullets establishing your strongest empirical pillars "
            "across distinct dimensions (economic, environmental, structural, and practical)."
        )
    elif current_round < max_rounds:
        phase_goal = (
            f"Clash & Expansion (Round {current_round}) — Dedicate your first 2 to 3 bullets to dismantling the opponent's "
            "previous turn, and dedicate your final 1 to 2 bullets to opening a brand-new, unaddressed empirical angle."
        )
    else:
        phase_goal = (
            f"Final Closing Clash (Round {current_round}) — Dedicate your first 2 to 3 bullets to refuting the opponent's "
            "latest claims, and dedicate your final 1 to 2 bullets to proving why your side wins the macro big picture."
        )

    populated = (
        template
        .replace("{topic}", topic)
        .replace("{current_round}", str(current_round))
        .replace("{max_rounds}", str(max_rounds))
        .replace("{phase_goal}", phase_goal)
    )

    return f"{populated}\n\n--- LIVE DEBATE CONTEXT ---\nProposition: \"{topic}\"\nCurrent Round: {current_round} of {max_rounds}\nRound Objective: {phase_goal}"

def enforce_wait_time(last_call_ts: float, wait_sec: int, next_agent_label: str):
    """Strictly pauses execution between EACH AI agent turn if clicked faster than wait_sec."""
    if last_call_ts > 0 and wait_sec > 0:
        elapsed = time.time() - last_call_ts
        remaining = wait_sec - elapsed
        if remaining > 0:
            gr.Info(f"⏳ Pacing API limits before {next_agent_label}: waiting {remaining:.1f}s...")
            time.sleep(remaining)

def call_with_rate_limit_retry(api_func, provider_name: str, max_retries: int = 3) -> str:
    """Executes an API call and automatically backs off on 429 / TPM / Quota errors."""
    for attempt in range(max_retries):
        try:
            return api_func()
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = any(k in err_msg for k in ["429", "rate_limit", "rate limit", "quota", "resource_exhausted", "tpm"])
            if is_rate_limit and attempt < max_retries - 1:
                wait_seconds = 15 * (attempt + 1)
                gr.Warning(f"⚠️ {provider_name} TPM/RPM limit reached. Auto-waiting {wait_seconds}s before retry ({attempt + 1}/{max_retries - 1})...")
                time.sleep(wait_seconds)
            else:
                raise gr.Error(f"{provider_name} API Error: {str(e)}")

def call_groq_supporter(system_prompt: str, user_prompt: str, groq_model: str) -> str:
    """Calls Groq API (optimized for openai/gpt-oss-120b reasoning model and standard Llama models)."""
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise gr.Error("Missing GROQ_API_KEY in Hugging Face Space Secrets.")

    def _execute():
        client = Groq(api_key=groq_key)
        combined_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        
        kwargs = {
            "model": groq_model,
            "messages": [{"role": "user", "content": combined_prompt}],
            "temperature": 0.6,
            "max_completion_tokens": 2048,
        }
        if "gpt-oss" in groq_model:
            kwargs["reasoning_effort"] = "low"

        completion = client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty response. Try clicking Next Turn again.")
        return content

    return call_with_rate_limit_retry(_execute, "Groq")

def call_gemini_contender(system_prompt: str, user_prompt: str, gemini_model: str, temperature: float = 0.6) -> str:
    """Calls Gemini API with dedicated thinking budget + ample output token headroom."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        raise gr.Error("Missing GEMINI_API_KEY in Hugging Face Space Secrets.")

    def _execute():
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model=gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=1024
                ),
            )
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response. Try clicking Next Turn again.")
        return response.text

    return call_with_rate_limit_retry(_execute, "Gemini")

def build_tpm_safe_transcript(history: list) -> str:
    """Keeps Round 1 (foundational scope) + the last 4 turns verbatim to prevent TPM bloat on long loops."""
    if len(history) <= 6:
        selected = history
    else:
        selected = (
            history[:2]
            + [{"role": "System", "round": "...", "content": "*[Earlier middle rounds omitted to preserve token budget]*"}]
            + history[-4:]
        )
    return "\n\n".join(
        [f"[Round {m['round']} - {m['role']}]:\n{m['content']}" for m in selected]
    )

def render_feed_html(topic: str, history: list, max_loops: int, verdict_text: str = "") -> str:
    """Renders the scrollable stack of Supporter, Contender, and optional Final Verdict cards."""
    if not history:
        return "<div style='padding:20px; text-align:center; color:#6b7280;'>Enter a proposition above and click <b>Start Debate</b> to begin Round 1.</div>"

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

    if verdict_text:
        verdict_html = markdown.markdown(verdict_text)
        html_blocks.append(f"""
        <div class="judge-card">
            <div class="judge-header">
                <span>⚖️ Neutral Adjudicator • Final Debate Verdict</span>
                <span class="round-badge">Official Evaluation</span>
            </div>
            <div class="card-body">{verdict_html}</div>
        </div>
        """)

    return "\n".join(html_blocks)

# ==========================================
# PDF TEXT SANITIZER & TRANSCRIPT EXPORTER
# ==========================================
def sanitize_for_pdf(text: str) -> str:
    """Normalizes subscripts/superscripts (e.g. CO₂ -> CO2) and converts Unicode punctuation for xhtml2pdf."""
    if not text:
        return ""

    # 1. Explicit punctuation & currency replacements before NFKD decomposition
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

    # 2. 1-Line NFKD Compatibility Decomposition: automatically turns CO₂ -> CO2, H₂O -> H2O, 10⁶ -> 106, ½ -> 1/2
    text = unicodedata.normalize("NFKD", text)

    # 3. Guarantee only Latin-1 printable characters reach xhtml2pdf's built-in Helvetica font
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def export_transcript_file(state):
    """Generates a styled, paginated PDF file containing the full debate history and optional verdict."""
    if not state or not state.get("history"):
        raise gr.Error("No debate content to export yet! Run at least one turn first.")

    raw_topic = state.get("topic", "Debate")
    safe_topic = sanitize_for_pdf(raw_topic)
    clean_topic = re.sub(r'[^a-zA-Z0-9_\- ]', '', safe_topic).strip().replace(" ", "_")[:40]
    if not clean_topic:
        clean_topic = "Transcript"
    filename = f"Debate_{clean_topic}.pdf"

    total_turns_done = state.get("current_turn", 0)
    rounds_completed = (total_turns_done + 1) // 2
    now_ist = datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S IST')

    # PDF Stylesheet
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
            line-height: 1.5;
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
            margin-bottom: 14px;
        }}
        .contender-card {{
            background-color: #fef2f2;
            border-left: 4px solid #dc2626;
            padding: 10px 14px;
            margin-bottom: 14px;
        }}
        .judge-card {{
            background-color: #fefce8;
            border-left: 4px solid #ca8a04;
            padding: 10px 14px;
            margin-top: 16px;
            margin-bottom: 14px;
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
        .header-judge {{
            font-weight: bold;
            color: #a16207;
            font-size: 10.5pt;
            margin-bottom: 6px;
        }}
        p, li {{
            margin-bottom: 4px;
        }}
    </style>
    </head>
    <body>
        <h1>Multi-Agent AI Debate Transcript</h1>
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

    if state.get("verdict"):
        safe_verdict = sanitize_for_pdf(state["verdict"])
        verdict_body_html = markdown.markdown(safe_verdict)
        pdf_html += f"""
        <div class="judge-card">
            <div class="header-judge">Neutral Adjudicator - Final Debate Verdict</div>
            {verdict_body_html}
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

# ==========================================
# ON-DEMAND NEUTRAL ADJUDICATOR VERDICT
# ==========================================
def run_judge_verdict(gemini_model, state):
    """Generates an unbiased, data-driven final verdict on demand (zero search credits consumed)."""
    if not state or len(state.get("history", [])) < 2:
        raise gr.Error("Please complete at least Round 1 (both Supporter and Contender) before generating a verdict.")

    if state.get("verdict"):
        transcript_path = export_transcript_file(state)
        return (
            state,
            render_feed_html(state["topic"], state["history"], state["max_loops"], state["verdict"]),
            gr.update(value="⚖️ Verdict Generated", interactive=False),
            gr.update(value=transcript_path, visible=True)
        )

    judge_system_prompt = (
        "You are an impartial, elite Policy & Empirical Debate Adjudicator. "
        "Evaluate the debate transcript objectively without bias or appeasement. "
        "Keep your response concise (160 to 220 words total) in an elevated analytical style:\n"
        "1. **Clash Assessment (1 short paragraph, 2-3 sentences):** Summarize the core tension and identify where the sharpest empirical clash occurred.\n"
        "2. **Scorecard (3 bullet points):**\n"
        "   - **Empirical Rigor & Citations:** Compare how effectively each side used verified statistics vs. generalizations.\n"
        "   - **Rebuttal & Clash Execution:** Identify who directly dismantled the opponent's points vs. who dodged key risks.\n"
        "   - **Real-World Feasibility:** Weigh the practical, economic, and structural realism of both arguments.\n"
        "3. **Final Verdict (1 short paragraph, 2-3 sentences):** Declare the winning side (Supporter or Contender) and state the decisive reason why their case prevailed."
    )

    transcript = "\n\n".join(
        [f"[Round {m['round']} - {m['role']} ({m['model']})]:\n{m['content']}" for m in state["history"]]
    )
    user_prompt = f"Proposition: \"{state['topic']}\"\n\nFull Debate Transcript:\n{transcript}\n\nDeliver your official Adjudicator Verdict."

    verdict_raw = call_gemini_contender(judge_system_prompt, user_prompt, gemini_model, temperature=0.3)
    state["verdict"] = clean_agent_output(verdict_raw)

    transcript_path = export_transcript_file(state)

    return (
        state,
        render_feed_html(state["topic"], state["history"], state["max_loops"], state["verdict"]),
        gr.update(value="⚖️ Verdict Generated", interactive=False, visible=True),
        gr.update(value=transcript_path, visible=True)
    )

def run_debate_step(topic_input, max_loops_input, cooldown_input, groq_model, gemini_model, state):
    """Executes a single step-by-step turn with daily quota protection, dynamic Tavily+Exa search, and PDF export."""
    if not topic_input or not topic_input.strip():
        raise gr.Error("Please enter a debate topic or proposition first.")

    is_first_turn_of_debate = not state["active"]

    # 1. Check Daily Limit BEFORE starting a brand-new debate (Turn 1)
    if is_first_turn_of_debate:
        allowed, used_today = can_start_new_debate()
        if not allowed:
            raise gr.Error(
                f"⏳ Daily limit of {MAX_DAILY_DEBATES} debates reached ({used_today}/{MAX_DAILY_DEBATES} used today) "
                f"to protect free-tier API quotas. Quota resets at midnight IST!"
            )

    turn_idx = state["current_turn"]
    max_loops_val = int(max_loops_input) if is_first_turn_of_debate else state["max_loops"]
    total_turns = max_loops_val * 2
    current_topic = topic_input.strip() if is_first_turn_of_debate else state["topic"]

    if turn_idx >= total_turns:
        transcript_path = export_transcript_file(state)
        verdict_btn_state = (
            gr.update(value="⚖️ Verdict Generated", interactive=False, visible=True)
            if state.get("verdict")
            else gr.update(value="⚖️ Generate Final Verdict", interactive=True, visible=True)
        )
        return (
            state,
            render_feed_html(state["topic"], state["history"], state["max_loops"], state.get("verdict", "")),
            gr.update(value="🏁 Debate Concluded", interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=transcript_path, visible=True),
            verdict_btn_state
        )

    current_round = (turn_idx // 2) + 1
    agent_role = "Supporter" if (turn_idx % 2 == 0) else "Contender"
    is_very_first_speech = (turn_idx == 0)

    # Enforce wait time before EVERY agent execution (after the very first opening turn)
    wait_sec = int(cooldown_input)
    enforce_wait_time(state.get("last_agent_ts", 0.0), wait_sec, f"{agent_role} (Round {current_round})")

    # Extract opponent's last speech so Round 2+ search pulls targeted counter-evidence
    last_opponent_text = state["history"][-1]["content"] if state["history"] else ""

    # Fetch live verified facts & figures from BOTH Tavily and Exa with dynamic round lens
    web_facts = fetch_combined_research(current_topic, agent_role, current_round, last_opponent_text)

    system_prompt = load_system_prompt(
        agent_role=agent_role,
        topic=current_topic,
        current_round=current_round,
        max_rounds=max_loops_val,
        is_very_first_speech=is_very_first_speech
    )

    research_block = (
        f"VERIFIED LIVE WEB RESEARCH (Round {current_round} Tavily + Exa Neural Search):\n{web_facts}\n\n"
        f"CRITICAL RULES FOR THIS TURN:\n"
        f"1. Cite actual organization, journal, or publication names (e.g., NITI Aayog, ICRIER, World Bank). NEVER output '(Source 1)' or '(Source 2)'.\n"
        f"2. Start every bullet directly with its bolded analytical point (e.g., '**Lifecycle emissions gap:** ...'). NEVER start a bullet with labels like 'Rebuttal:', 'Direct Rebuttal:', or 'New Front:'.\n"
        f"3. ANTI-FABRICATION RULE: Never invent convenient counter-numbers to refute an opponent's statistic. If an exact counter-stat is not in the research, attack the methodology, lifecycle boundary, or structural trade-off using verified data."
    )

    if not state["history"]:
        user_prompt = f"{research_block}\n\nPlease open the debate on the proposition: '{current_topic}'."
    else:
        transcript = build_tpm_safe_transcript(state["history"])
        user_prompt = (
            f"Here is the debate transcript so far:\n\n{transcript}\n\n"
            f"{research_block}\n\n"
            f"Now deliver your Round {current_round} response as the {agent_role}. Do not repeat studies or arguments you already used in earlier rounds."
        )

    if agent_role == "Supporter":
        raw_reply = call_groq_supporter(system_prompt, user_prompt, groq_model)
        used_model = groq_model
    else:
        raw_reply = call_gemini_contender(system_prompt, user_prompt, gemini_model)
        used_model = gemini_model

    reply_text = clean_agent_output(raw_reply)

    # 2. Increment Daily Counter ONLY after Turn 1 succeeds
    if is_first_turn_of_debate:
        new_daily_count = record_new_debate_started()
        state["topic"] = current_topic
        state["max_loops"] = max_loops_val
        state["active"] = True
        gr.Info(f"✅ Debate started ({new_daily_count} of {MAX_DAILY_DEBATES} daily debates used today).")

    # Record completion timestamp so the next agent turn strictly waits wait_sec
    state["last_agent_ts"] = time.time()

    state["history"].append({
        "role": agent_role,
        "round": current_round,
        "model": used_model,
        "content": reply_text
    })
    state["current_turn"] += 1

    # Also log to Hugging Face container console for live monitoring under the "Logs" tab
    print(f"\n{'='*50}\n[DEBATE TURN COMPLETED] Topic: {state['topic']} | Round {current_round} | {agent_role} ({used_model})\n{'='*50}")

    new_turn_idx = state["current_turn"]
    if new_turn_idx < total_turns:
        next_round = (new_turn_idx // 2) + 1
        if new_turn_idx % 2 == 0:
            btn_label = f"🟢 Next Turn: Go to Supporter [Groq] (Round {next_round} of {state['max_loops']})"
        else:
            btn_label = f"🔴 Next Turn: Go to Contender [Gemini] (Round {next_round} of {state['max_loops']})"
        btn_interactive = True
    else:
        btn_label = f"🏁 Debate Complete ({state['max_loops']} Rounds Finished) — Click Reset for New Topic"
        btn_interactive = False

    # Prepare updated PDF file so DownloadButton works immediately on click
    transcript_path = export_transcript_file(state)

    # Show the On-Demand Verdict button once at least Round 1 is completed (2 turns)
    show_verdict_btn = (new_turn_idx >= 2)
    verdict_btn_update = gr.update(
        value="⚖️ Generate Final Verdict",
        visible=show_verdict_btn,
        interactive=(not bool(state.get("verdict")))
    )

    return (
        state,
        render_feed_html(state["topic"], state["history"], state["max_loops"], state.get("verdict", "")),
        gr.update(value=btn_label, interactive=btn_interactive),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=transcript_path, visible=True),
        verdict_btn_update
    )

def reset_debate():
    fresh_state = {
        "history": [],
        "current_turn": 0,
        "active": False,
        "topic": "",
        "max_loops": DEFAULT_MAX_LOOPS,
        "last_agent_ts": 0.0,
        "verdict": ""
    }
    return (
        fresh_state,
        render_feed_html("", [], DEFAULT_MAX_LOOPS, ""),
        gr.update(value="🚀 Start Debate (Trigger Round 1: Supporter [Groq])", interactive=True),
        gr.update(value="", interactive=True),
        gr.update(value=DEFAULT_MAX_LOOPS, interactive=True),
        gr.update(value=None, visible=False),
        gr.update(value="⚖️ Generate Final Verdict", visible=False, interactive=True)
    )

# ==========================================
# PHASE 3: GRADIO INTERFACE (WITH PASSWORD GATE)
# ==========================================
with gr.Blocks(css=CUSTOM_CSS, title="Multi-Agent AI Debate Engine") as demo:
    debate_state = gr.State({
        "history": [],
        "current_turn": 0,
        "active": False,
        "topic": "",
        "max_loops": DEFAULT_MAX_LOOPS,
        "last_agent_ts": 0.0,
        "verdict": ""
    })

    gr.Markdown("# ⚖️ The Multi-Agent AI Debate Engine\n*🟢 Supporter (Groq `openai/gpt-oss-120b`) vs. 🔴 Contender (Google `gemini-2.5-flash`) • Grounded with Live Tavily + Exa Neural Search*")

    # ------------------------------------------
    # 1. PASSWORD GATE SCREEN (SHOWN FIRST)
    # ------------------------------------------
    with gr.Column(visible=True) as login_view:
        gr.Markdown("### 🔒 Protected Access\nPlease enter the access password to unlock the Debate Engine.")
        with gr.Row():
            password_input = gr.Textbox(
                label="Access Password",
                type="password",
                placeholder="Enter password...",
                scale=3
            )
            unlock_btn = gr.Button("🔓 Unlock Debate Engine", variant="primary", scale=1)
        login_error_msg = gr.Markdown("")

    # ------------------------------------------
    # 2. MAIN DEBATE ENGINE (HIDDEN UNTIL UNLOCKED)
    # ------------------------------------------
    with gr.Column(visible=False) as main_debate_view:
        with gr.Accordion("⚙️ Debate Settings (Loops, Wait Time & Model Selection)", open=True):
            with gr.Row():
                loops_slider = gr.Slider(
                    minimum=1,
                    maximum=DEFAULT_MAX_LOOPS,
                    value=DEFAULT_MAX_LOOPS,
                    step=1,
                    label=f"Number of Debate Rounds / Loops (Max {DEFAULT_MAX_LOOPS})",
                    scale=2
                )
                cooldown_slider = gr.Slider(
                    minimum=0,
                    maximum=60,
                    value=DEFAULT_WAIT_TIME_SEC,
                    step=1,
                    label="Wait Time Between Each Agent Execution (Seconds)",
                    scale=2
                )
            with gr.Row():
                groq_model_dropdown = gr.Dropdown(
                    choices=["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                    value="openai/gpt-oss-120b",
                    label="🟢 Supporter Model (Groq)",
                    scale=1
                )
                gemini_model_dropdown = gr.Dropdown(
                    choices=["gemini-2.5-flash", "gemini-2.5-pro"],
                    value="gemini-2.5-flash",
                    label="🔴 Contender / Adjudicator Model (Gemini)",
                    scale=1
                )

        with gr.Row():
            topic_box = gr.Textbox(
                label="Debate Proposition / Topic",
                placeholder="e.g., Is social media positively impacting society?",
                scale=4
            )
            reset_btn = gr.Button("🔄 Reset / New Debate", variant="secondary", scale=1)

        feed_html = gr.HTML(value=render_feed_html("", [], DEFAULT_MAX_LOOPS, ""))

        with gr.Row():
            next_turn_btn = gr.Button(
                "🚀 Start Debate (Trigger Round 1: Supporter [Groq])",
                variant="primary",
                size="lg",
                scale=3
            )
            verdict_btn = gr.Button(
                "⚖️ Generate Final Verdict",
                variant="secondary",
                size="lg",
                visible=False,
                scale=1
            )
            export_btn = gr.DownloadButton(
                "📥 Download Transcript (.pdf)",
                variant="secondary",
                size="lg",
                visible=False,
                scale=1
            )

    # Wire up Password Gate (supports clicking the button OR pressing Enter)
    unlock_btn.click(
        fn=verify_password,
        inputs=[password_input],
        outputs=[login_view, main_debate_view, login_error_msg]
    )
    password_input.submit(
        fn=verify_password,
        inputs=[password_input],
        outputs=[login_view, main_debate_view, login_error_msg]
    )

    # Wire up Main Debate Controls
    next_turn_btn.click(
        fn=run_debate_step,
        inputs=[topic_box, loops_slider, cooldown_slider, groq_model_dropdown, gemini_model_dropdown, debate_state],
        outputs=[debate_state, feed_html, next_turn_btn, topic_box, loops_slider, export_btn, verdict_btn]
    )

    verdict_btn.click(
        fn=run_judge_verdict,
        inputs=[gemini_model_dropdown, debate_state],
        outputs=[debate_state, feed_html, verdict_btn, export_btn]
    )

    export_btn.click(
        fn=export_transcript_file,
        inputs=[debate_state],
        outputs=[export_btn]
    )

    reset_btn.click(
        fn=reset_debate,
        inputs=[],
        outputs=[debate_state, feed_html, next_turn_btn, topic_box, loops_slider, export_btn, verdict_btn]
    )

if __name__ == "__main__":
    demo.launch()