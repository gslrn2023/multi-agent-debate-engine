import os
import time
import markdown
import gradio as gr
from groq import Groq
from google import genai
from google.genai import types

# ==========================================
# PROMPT FILE NAMES (TWO SEPARATE FILES)
# ==========================================
SUPPORTER_PROMPT_FILE = "supporter_prompt.txt"
CONTENDER_PROMPT_FILE = "contender_prompt.txt"

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

    # Safely replace placeholders if present in the text file, and append live round context
    populated = (
        template
        .replace("{topic}", topic)
        .replace("{current_round}", str(current_round))
        .replace("{max_rounds}", str(max_rounds))
        .replace("{phase_goal}", phase_goal)
    )

    return f"{populated}\n\n--- LIVE DEBATE CONTEXT ---\nProposition: \"{topic}\"\nCurrent Round: {current_round} of {max_rounds}\nRound Objective: {phase_goal}"

def enforce_cooldown(last_call_ts: float, cooldown_sec: int, provider_name: str):
    """Pauses execution if the time since the last call to the same provider is shorter than cooldown_sec."""
    if last_call_ts > 0 and cooldown_sec > 0:
        elapsed = time.time() - last_call_ts
        remaining = cooldown_sec - elapsed
        if remaining > 0:
            gr.Info(f"⏳ Pacing TPM/RPM limits for {provider_name}: waiting {remaining:.1f}s...")
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
            "max_completion_tokens": 1024,
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
    """Calls Gemini API (optimized for gemini-2.5-flash)."""
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
                temperature=0.7,
                max_output_tokens=1024,
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

def run_debate_step(topic_input, max_loops_input, cooldown_input, groq_model, gemini_model, state):
    """Executes a single step-by-step turn."""
    if not topic_input or not topic_input.strip():
        raise gr.Error("Please enter a debate topic or proposition first.")

    if not state["active"]:
        state["topic"] = topic_input.strip()
        state["max_loops"] = int(max_loops_input)
        state["active"] = True

    turn_idx = state["current_turn"]
    total_turns = state["max_loops"] * 2

    if turn_idx >= total_turns:
        return (
            state,
            render_feed_html(state["topic"], state["history"], state["max_loops"]),
            gr.update(value="🏁 Debate Concluded", interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

    current_round = (turn_idx // 2) + 1
    agent_role = "Supporter" if (turn_idx % 2 == 0) else "Contender"

    system_prompt = load_system_prompt(
        agent_role=agent_role,
        topic=state["topic"],
        current_round=current_round,
        max_rounds=state["max_loops"]
    )

    if not state["history"]:
        user_prompt = f"Please open the debate on the proposition: '{state['topic']}'."
    else:
        transcript = build_tpm_safe_transcript(state["history"])
        user_prompt = (
            f"Here is the debate transcript so far:\n\n{transcript}\n\n"
            f"Now deliver your Round {current_round} response as the {agent_role}."
        )

    cooldown_sec = int(cooldown_input)
    if agent_role == "Supporter":
        enforce_cooldown(state.get("last_groq_ts", 0.0), cooldown_sec, "Groq")
        reply_text = call_groq_supporter(system_prompt, user_prompt, groq_model)
        state["last_groq_ts"] = time.time()
        used_model = groq_model
    else:
        enforce_cooldown(state.get("last_gemini_ts", 0.0), cooldown_sec, "Gemini")
        reply_text = call_gemini_contender(system_prompt, user_prompt, gemini_model)
        state["last_gemini_ts"] = time.time()
        used_model = gemini_model

    state["history"].append({
        "role": agent_role,
        "round": current_round,
        "model": used_model,
        "content": reply_text
    })
    state["current_turn"] += 1

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

    return (
        state,
        render_feed_html(state["topic"], state["history"], state["max_loops"]),
        gr.update(value=btn_label, interactive=btn_interactive),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )

def reset_debate():
    fresh_state = {
        "history": [],
        "current_turn": 0,
        "active": False,
        "topic": "",
        "max_loops": 3,
        "last_groq_ts": 0.0,
        "last_gemini_ts": 0.0
    }
    return (
        fresh_state,
        render_feed_html("", [], 3),
        gr.update(value="🚀 Start Debate (Trigger Round 1: Supporter [Groq])", interactive=True),
        gr.update(value="", interactive=True),
        gr.update(value=3, interactive=True)
    )

# ==========================================
# PHASE 3: GRADIO INTERFACE
# ==========================================
with gr.Blocks(css=CUSTOM_CSS, title="Multi-Agent AI Debate Engine") as demo:
    debate_state = gr.State({
        "history": [],
        "current_turn": 0,
        "active": False,
        "topic": "",
        "max_loops": 3,
        "last_groq_ts": 0.0,
        "last_gemini_ts": 0.0
    })

    gr.Markdown("# ⚖️ The Multi-Agent AI Debate Engine\n*🟢 Supporter (Groq `openai/gpt-oss-120b`) vs. 🔴 Contender (Google `gemini-2.5-flash`)*")

    with gr.Accordion("⚙️ Debate Settings (Loops, TPM Cooldown & Model Selection)", open=True):
        with gr.Row():
            loops_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=3,
                step=1,
                label="Number of Debate Rounds / Loops (1 Loop = Supporter + Contender)",
                scale=2
            )
            cooldown_slider = gr.Slider(
                minimum=0,
                maximum=20,
                value=6,
                step=1,
                label="TPM/RPM Safety Wait Time (Seconds between same-model calls)",
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

    feed_html = gr.HTML(value=render_feed_html("", [], 3))

    next_turn_btn = gr.Button(
        "🚀 Start Debate (Trigger Round 1: Supporter [Groq])",
        variant="primary",
        size="lg"
    )

    next_turn_btn.click(
        fn=run_debate_step,
        inputs=[topic_box, loops_slider, cooldown_slider, groq_model_dropdown, gemini_model_dropdown, debate_state],
        outputs=[debate_state, feed_html, next_turn_btn, topic_box, loops_slider]
    )

    reset_btn.click(
        fn=reset_debate,
        inputs=[],
        outputs=[debate_state, feed_html, next_turn_btn, topic_box, loops_slider]
    )

if __name__ == "__main__":
    demo.launch()