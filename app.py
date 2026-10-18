import os
import re
import json
import time
import tempfile
import threading
from datetime import datetime, timedelta, timezone
import markdown
import gradio as gr
import spaces
from groq import Groq
from google import genai
from google.genai import types
from tavily import TavilyClient
from xhtml2pdf import pisa

# ==========================================
# CONFIGURABLE VARIABLES (LOOPS, WAIT & DAILY LIMIT)
# ==========================================
DEFAULT_MAX_LOOPS = 3       # // adjust the value here // (Default & Max number of debate loops/rounds)
DEFAULT_WAIT_TIME_SEC = 30  # // adjust the value here // (Wait time in seconds between EACH AI agent execution)
MAX_DAILY_DEBATES = 4       # // adjust the value here // (Global max debates per day across all users)
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
# CUSTOM CSS FOR DISTINCT AGENT PALETTES
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
# TAVILY LIVE WEB FACT RETRIEVAL
# ==========================================
def fetch_tavily_facts(topic: str, agent_role: str, current_round: int) -> str:
    """Queries Tavily for verified facts, statistics, and references tailored to the agent's stance."""
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not tavily_key:
        raise gr.Error("Missing TAVILY_API_KEY in Hugging Face Space Secrets.")

    if agent_role == "Supporter":
        focus = "empirical evidence statistics benefits success cases data supporting" if current_round == 1 else "verified statistics empirical studies data refuting criticisms of"
    else:
        focus = "empirical criticisms risks negative impacts statistics failure cases against" if current_round == 1 else "counter-evidence data studies structural flaws risks of"

    # Keep query under 380 characters (Tavily max query length is 400 chars)
    query = f"{topic[:250]} {focus}"

    try:
        client = TavilyClient(api_key=tavily_key)
        response = client.search(
            query=query,
            search_depth="basic",  # Uses 1 credit per search (1,000 free/month)
            max_results=4,
            include_answer=True
        )

        snippets = []
        if response.get("answer"):
            snippets.append(f"- **Web Summary:** {response['answer'][:350]}")

        for idx, res in enumerate(response.get("results", []), 1):
            title = res.get("title", "Verified Source")
            url = res.get("url", "")
            # Trim content to 350 chars per result to protect Groq TPM limits
            content = res.get("content", "").strip().replace("\n", " ")[:350]
            snippets.append(f"- **Source {idx} ({title} | {url}):** {content}...")

        if not snippets:
            return "No external web snippets returned; rely on verified historical and empirical data."

        return "\n".join(snippets)
    except Exception as e:
        gr.Warning(f"⚠️ Tavily search warning: {str(e)}. Proceeding with internal knowledge.")
        return "Web search temporarily unavailable; rely on verified empirical and historical knowledge."

# ==========================================
# PHASE 2: EXTERNAL PROMPT LOADER & API CALLS
# ==========================================
def load_system_prompt(agent_role: str, topic: str, current_round: int, max_rounds: int) -> str:
    """Loads the prompt from supporter_prompt.txt or contender_prompt.txt and injects round variables."""
    file_path = SUPPORTER_PROMPT_FILE if agent_role == "Supporter" else CONTENDER_PROMPT_FILE

    if not os.path.exists(file_path):
        raise gr.Error(f"Missing prompt file: '{file_path}'. Please create it in your Hugging Face Space Files tab.")

    with open(file_path, "r", encoding="utf-8") as f:
        template = f.read()

    if current_round == 1:
        phase_goal = "Opening Round — Establish clear definitions, scope, and your foundational case."
    elif current_round < max_rounds:
        phase_goal = "Middle Clash & Rebuttal Round — Directly dismantle the opponent's last arguments, test their assumptions, and advance the clash."
    else:
        phase_goal = "Final Closing Round — Introduce no new arguments; summarize key clashes, weigh final impacts, and deliver a decisive closing conclusion."

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
            "temperature": 0.65,
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

def call_gemini_contender(system_prompt: str, user_prompt: str, gemini_model: str) -> str:
    """Calls Gemini API with dedicated thinking budget + ample output token headroom for full rebuttals."""
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
                temperature=0.65,
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

def render_feed_html(topic: str, history: list, max_loops: int) -> str:
    """Renders the scrollable stack of Supporter and Contender cards."""
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
    return "\n".join(html_blocks)

# ==========================================
# PDF TEXT SANITIZER & TRANSCRIPT EXPORTER
# ==========================================
def sanitize_for_pdf(text: str) -> str:
    """Replaces Unicode hyphens, dashes, smart quotes, and symbols that cause black squares in xhtml2pdf."""
    if not text:
        return ""

    replacements = {
        "\u2010": "-",   # Hyphen
        "\u2011": "-",   # Non-breaking hyphen (primary cause of black squares in gpt-oss-120b output)
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
        "\u20b9": "Rs.", # Indian Rupee symbol (not in base Type1 Helvetica)
    }

    for unicode_char, ascii_equiv in replacements.items():
        text = text.replace(unicode_char, ascii_equiv)

    # Guarantee only Latin-1 printable characters reach xhtml2pdf's built-in Helvetica font
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def export_transcript_file(state):
    """Generates a styled, paginated PDF file containing the full debate history without missing-glyph boxes."""
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

    pdf_html += "</body></html>"

    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    
    with open(filepath, "w+b") as f:
        pisa_status = pisa.CreatePDF(pdf_html, dest=f)
        
    if pisa_status.err:
        raise gr.Error("PDF generation error occurred.")

    return filepath

def run_debate_step(topic_input, max_loops_input, cooldown_input, groq_model, gemini_model, state):
    """Executes a single step-by-step turn with daily quota protection, live Tavily search, and PDF export."""
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
        return (
            state,
            render_feed_html(state["topic"], state["history"], state["max_loops"]),
            gr.update(value="🏁 Debate Concluded", interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=transcript_path, visible=True)
        )

    current_round = (turn_idx // 2) + 1
    agent_role = "Supporter" if (turn_idx % 2 == 0) else "Contender"

    # Enforce wait time before EVERY agent execution (after the very first opening turn)
    wait_sec = int(cooldown_input)
    enforce_wait_time(state.get("last_agent_ts", 0.0), wait_sec, f"{agent_role} (Round {current_round})")

    # Fetch live verified facts & figures from Tavily for this turn
    web_facts = fetch_tavily_facts(current_topic, agent_role, current_round)

    system_prompt = load_system_prompt(
        agent_role=agent_role,
        topic=current_topic,
        current_round=current_round,
        max_rounds=max_loops_val
    )

    research_block = (
        f"VERIFIED LIVE WEB RESEARCH (Tavily Search Results):\n{web_facts}\n\n"
        f"INSTRUCTION: Ground your arguments, statistics, figures, and source references directly in the verified web research above alongside established empirical facts."
    )

    if not state["history"]:
        user_prompt = f"{research_block}\n\nPlease open the debate on the proposition: '{current_topic}'."
    else:
        transcript = build_tpm_safe_transcript(state["history"])
        user_prompt = (
            f"Here is the debate transcript so far:\n\n{transcript}\n\n"
            f"{research_block}\n\n"
            f"Now deliver your Round {current_round} response as the {agent_role}."
        )

    if agent_role == "Supporter":
        reply_text = call_groq_supporter(system_prompt, user_prompt, groq_model)
        used_model = groq_model
    else:
        reply_text = call_gemini_contender(system_prompt, user_prompt, gemini_model)
        used_model = gemini_model

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

    return (
        state,
        render_feed_html(state["topic"], state["history"], state["max_loops"]),
        gr.update(value=btn_label, interactive=btn_interactive),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=transcript_path, visible=True)
    )

def reset_debate():
    fresh_state = {
        "history": [],
        "current_turn": 0,
        "active": False,
        "topic": "",
        "max_loops": DEFAULT_MAX_LOOPS,
        "last_agent_ts": 0.0
    }
    return (
        fresh_state,
        render_feed_html("", [], DEFAULT_MAX_LOOPS),
        gr.update(value="🚀 Start Debate (Trigger Round 1: Supporter [Groq])", interactive=True),
        gr.update(value="", interactive=True),
        gr.update(value=DEFAULT_MAX_LOOPS, interactive=True),
        gr.update(value=None, visible=False)
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
        "last_agent_ts": 0.0
    })

    gr.Markdown("# ⚖️ The Multi-Agent AI Debate Engine\n*🟢 Supporter (Groq `openai/gpt-oss-120b`) vs. 🔴 Contender (Google `gemini-2.5-flash`) • Grounded with Live Tavily Web Search*")

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
                    label="🔴 Contender Model (Gemini)",
                    scale=1
                )

        with gr.Row():
            topic_box = gr.Textbox(
                label="Debate Proposition / Topic",
                placeholder="e.g., Is social media positively impacting society?",
                scale=4
            )
            reset_btn = gr.Button("🔄 Reset / New Debate", variant="secondary", scale=1)

        feed_html = gr.HTML(value=render_feed_html("", [], DEFAULT_MAX_LOOPS))

        with gr.Row():
            next_turn_btn = gr.Button(
                "🚀 Start Debate (Trigger Round 1: Supporter [Groq])",
                variant="primary",
                size="lg",
                scale=3
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
        outputs=[debate_state, feed_html, next_turn_btn, topic_box, loops_slider, export_btn]
    )

    export_btn.click(
        fn=export_transcript_file,
        inputs=[debate_state],
        outputs=[export_btn]
    )

    reset_btn.click(
        fn=reset_debate,
        inputs=[],
        outputs=[debate_state, feed_html, next_turn_btn, topic_box, loops_slider, export_btn]
    )

if __name__ == "__main__":
    demo.launch()