import os
import re
import time
import tempfile
import markdown
import gradio as gr
import spaces
from groq import Groq
from google import genai
from google.genai import types
from tavily import TavilyClient

# ==========================================
# CONFIGURABLE VARIABLES (LOOPS & WAIT TIME)
# ==========================================
DEFAULT_MAX_LOOPS = 3       # // adjust the value here // (Default & Max number of debate loops/rounds)
DEFAULT_WAIT_TIME_SEC = 30   # // adjust the value here // (Wait time in seconds between EACH AI agent execution)

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
# TRANSCRIPT EXPORTER HELPER
# ==========================================
def export_transcript_file(state):
    """Generates a clean, formatted Markdown file containing the full debate history."""
    if not state or not state.get("history"):
        raise gr.Error("No debate content to export yet! Run at least one turn first.")

    topic = state.get("topic", "Debate")
    clean_topic = re.sub(r'[^a-zA-Z0-9_\- ]', '', topic).strip().replace(" ", "_")[:40]
    if not clean_topic:
        clean_topic = "Transcript"
    filename = f"Debate_{clean_topic}.md"

    total_turns_done = state.get("current_turn", 0)
    rounds_completed = (total_turns_done + 1) // 2

    lines = [
        f"# ⚖️ Multi-Agent AI Debate Transcript",
        f"**Proposition:** {topic}",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Progress:** {total_turns_done} turns completed (Round {rounds_completed} of {state.get('max_loops', DEFAULT_MAX_LOOPS)})",
        f"\n---\n"
    ]

    for item in state["history"]:
        role_icon = "🟢" if item["role"] == "Supporter" else "🔴"
        lines.append(f"### {role_icon} Round {item['round']} — {item['role']} ({item['model']})\n")
        lines.append(f"{item['content'].strip()}\n")
        lines.append("\n---\n")

    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath

def run_debate_step(topic_input, max_loops_input, cooldown_input, groq_model, gemini_model, state):
    """Executes a single step-by-step turn with live Tavily search, strict wait time, and transcript export update."""
    if not topic_input or not topic_input.strip():
        raise gr.Error("Please enter a debate topic or proposition first.")

    if not state["active"]:
        state["topic"] = topic_input.strip()
        state["max_loops"] = int(max_loops_input)
        state["active"] = True

    turn_idx = state["current_turn"]
    total_turns = state["max_loops"] * 2

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
    web_facts = fetch_tavily_facts(state["topic"], agent_role, current_round)

    system_prompt = load_system_prompt(
        agent_role=agent_role,
        topic=state["topic"],
        current_round=current_round,
        max_rounds=state["max_loops"]
    )

    research_block = (
        f"VERIFIED LIVE WEB RESEARCH (Tavily Search Results):\n{web_facts}\n\n"
        f"INSTRUCTION: Ground your arguments, statistics, figures, and source references directly in the verified web research above alongside established empirical facts."
    )

    if not state["history"]:
        user_prompt = f"{research_block}\n\nPlease open the debate on the proposition: '{state['topic']}'."
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

    # Prepare updated Markdown file so DownloadButton works immediately on click
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
                "📥 Download Transcript (.md)",
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