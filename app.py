# ==============================================================================
# MODULE: app.py (Main Entry Point)
# Version: v2.9.3 | Date: 2026-11-03
# Purpose: Main step-by-step debate orchestrator and Gradio UI assembly.
# ==============================================================================

import time
from datetime import datetime
import gradio as gr

from config import (
    APP_VERSION,
    TRACE,
    DEFAULT_MAX_LOOPS,
    DEFAULT_WAIT_TIME_SEC,
    MAX_DAILY_DEBATES,
    IST_TZ,
    CUSTOM_CSS,
    zerogpu_healthcheck,
    can_start_new_debate,
    record_new_debate_started,
    verify_password
)
from llm_agents import (
    load_system_prompt,
    enforce_wait_time,
    call_groq_supporter,
    call_gemini_contender,
    build_tpm_safe_transcript,
    clean_agent_output
)
from search_engine import fetch_combined_research
from exporters import (
    render_feed_html,
    export_transcript_file,
    export_trace_json_file
)

def run_debate_step(topic_input, max_loops_input, cooldown_input, groq_model, gemini_model, state):
    """Executes a single step-by-step turn with strict provider isolation, cross-round query memory, PDF export, and TRACE logging."""
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
        trace_path = export_trace_json_file(state) if TRACE else None
        return (
            state,
            render_feed_html(state["topic"], state["history"], state["max_loops"]),
            gr.update(value="🏁 Debate Concluded", interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=transcript_path, visible=True),
            gr.update(value=trace_path, visible=bool(TRACE and trace_path))
        )

    current_round = (turn_idx // 2) + 1
    agent_role = "Supporter" if (turn_idx % 2 == 0) else "Contender"
    active_model = groq_model if agent_role == "Supporter" else gemini_model
    is_very_first_speech = (turn_idx == 0)

    # Enforce wait time before EVERY agent execution (after the very first opening turn)
    wait_sec = int(cooldown_input)
    enforce_wait_time(state.get("last_agent_ts", 0.0), wait_sec, f"{agent_role} (Round {current_round})")

    # Extract opponent's last speech text and this agent's past queries to prevent query repetition
    last_opponent_text = state["history"][-1]["content"] if state["history"] else ""
    past_agent_queries = [
        entry.get("tavily_dispatched_query", "")
        for entry in state.get("trace_logs", [])
        if entry.get("role") == agent_role and entry.get("tavily_dispatched_query")
    ]

    # Fetch deduplicated facts & figures via Strictly Isolated Agentic Query Planner + Tavily + Exa
    web_facts, trace_meta, updated_seen_urls, updated_seen_titles = fetch_combined_research(
        topic=current_topic,
        agent_role=agent_role,
        current_round=current_round,
        active_model=active_model,
        last_opponent_text=last_opponent_text,
        seen_urls=state.get("seen_urls", []),
        seen_titles=state.get("seen_titles", []),
        past_agent_queries=past_agent_queries
    )
    state["seen_urls"] = updated_seen_urls
    state["seen_titles"] = updated_seen_titles

    system_prompt = load_system_prompt(
        agent_role=agent_role,
        topic=current_topic,
        current_round=current_round,
        max_rounds=max_loops_val,
        is_very_first_speech=is_very_first_speech
    )
    research_block = (
        f"VERIFIED LIVE WEB RESEARCH (Round {current_round} — Inferred Domain: {trace_meta['inferred_domain']}):\n{web_facts}\n\n"
        f"UNIVERSAL RULES FOR THIS TURN:\n"
        f"1. HARD-HITTING, EVIDENCE-DENSE FORMAT (180-230 WORDS TOTAL): Deliver a sharp 2-sentence opening hook, followed by 4 to 5 bullet points (30-40 words each), and a decisive 2-sentence closing punchline. NEVER print section headers like 'Opening Hook', 'Fact-Sheet', or 'Closing Takeaway'.\n"
        f"2. STRICT NUMERICAL FIDELITY & QUALITATIVE PARITY: Only write a number, percentage, or ratio if that exact figure is explicitly printed in the VERIFIED LIVE WEB RESEARCH above. When a snippet provides a qualitative, technical, or methodological finding without numbers, deploy that qualitative mechanism directly—NEVER invent or extrapolate percentages (such as fake '%', 'ha', or 'MMT' figures) to make a bullet sound quantitative.\n"
        f"3. MANDATORY CITATION ON EVERY BULLET: Every single bullet point (all 4 to 5 bullets) MUST end with a clean parenthetical citation naming the study, author, survey, or institution from the snippet header—e.g., (Author or Institution Name, Year). Never leave trailing bullets uncited, and never cite raw URLs.\n"
        f"4. MULTI-PRONG CLASH & NON-REPETITION: Directly audit the opponent's specific sub-claims from their last speech without repeating your own bullet headers or figures from earlier rounds."
    )
    
    if not state["history"]:
        user_prompt = f"{research_block}\n\nDeliver your Round 1 opening argument on: '{current_topic}'."
    else:
        transcript = build_tpm_safe_transcript(state["history"], current_agent_role=agent_role)
        user_prompt = (
            f"Debate transcript so far:\n\n{transcript}\n\n"
            f"{research_block}\n\n"
            f"Deliver your Round {current_round} response as the {agent_role} "
            f"(4 to 5 bullets: 2-3 direct rebuttals to the opponent's last speech + 1-2 brand-new evidence fronts not used in your earlier turns, 180-230 words total)."
        )

    if agent_role == "Supporter":
        raw_reply = call_groq_supporter(system_prompt, user_prompt, groq_model, temperature=0.20)
    else:
        raw_reply = call_gemini_contender(system_prompt, user_prompt, gemini_model, temperature=0.30)

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
        "model": active_model,
        "content": reply_text
    })

    # Record structured trace entry when TRACE is ON
    if TRACE:
        turn_trace_entry = {
            "turn_number": turn_idx + 1,
            "round": current_round,
            "role": agent_role,
            "model": active_model,
            "timestamp_ist": datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S IST'),
            "inferred_domain": trace_meta["inferred_domain"],
            "query_planner_source": trace_meta["query_planner_source"],
            "planner_errors": trace_meta.get("planner_errors", []),
            "compressed_topic_anchor": trace_meta["compressed_topic_anchor"],
            "extracted_opponent_keywords": trace_meta["extracted_opponent_keywords"],
            "past_queries_avoided": trace_meta.get("past_queries_avoided", []),
            "tavily_dispatched_query": trace_meta["tavily_dispatched_query"],
            "exa_dispatched_prompt": trace_meta["exa_dispatched_prompt"],
            "tavily_results": trace_meta["tavily_results"],
            "exa_results": trace_meta["exa_results"],
            "injected_research_block_chars": trace_meta["combined_research_char_length"],
            "total_unique_urls_seen": trace_meta["total_unique_urls_seen_so_far"],
            "total_unique_titles_seen": trace_meta["total_unique_titles_seen_so_far"],
            "raw_model_output": raw_reply,
            "cleaned_model_output": reply_text
        }
        state.setdefault("trace_logs", []).append(turn_trace_entry)
        print(f"\n{'='*60}\n[TRACE {APP_VERSION}] Round {current_round} | {agent_role} ({active_model})")
        print(f"  -> Inferred Domain   : {trace_meta['inferred_domain']} ({trace_meta['query_planner_source']})")
        if trace_meta.get("planner_errors"):
            print(f"  -> Planner Warnings  : {trace_meta['planner_errors']}")
        print(f"  -> Tavily Query      : {trace_meta['tavily_dispatched_query']}")
        print(f"  -> Exa Prompt        : {trace_meta['exa_dispatched_prompt']}")
        print(f"  -> Snippet Payload   : {trace_meta['combined_research_char_length']} chars | Unique URLs: {trace_meta['total_unique_urls_seen_so_far']}\n{'='*60}")
    else:
        print(f"\n{'='*50}\n[DEBATE TURN COMPLETED] Topic: {state['topic']} | Round {current_round} | {agent_role} ({active_model})\n{'='*50}")

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

    transcript_path = export_transcript_file(state)
    trace_path = export_trace_json_file(state) if TRACE else None

    return (
        state,
        render_feed_html(state["topic"], state["history"], state["max_loops"]),
        gr.update(value=btn_label, interactive=btn_interactive),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=transcript_path, visible=True),
        gr.update(value=trace_path, visible=bool(TRACE and trace_path))
    )

def reset_debate():
    fresh_state = {
        "history": [],
        "trace_logs": [],
        "seen_urls": [],
        "seen_titles": [],
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
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False)
    )

# ==========================================
# GRADIO INTERFACE (WITH PASSWORD GATE)
# ==========================================
with gr.Blocks(css=CUSTOM_CSS, title=f"Multi-Agent AI Debate Engine ({APP_VERSION})") as demo:
    debate_state = gr.State({
        "history": [],
        "trace_logs": [],
        "seen_urls": [],
        "seen_titles": [],
        "current_turn": 0,
        "active": False,
        "topic": "",
        "max_loops": DEFAULT_MAX_LOOPS,
        "last_agent_ts": 0.0
    })

    trace_badge = " • 🔍 `TRACE: ON`" if TRACE else ""
    gr.Markdown(
        f"# ⚖️ The Multi-Agent AI Debate Engine (`{APP_VERSION}`)\n"
        f"*🟢 Supporter (100% Groq `openai/gpt-oss-120b`) vs. 🔴 Contender (100% Google `gemini-2.5-flash`) • Isolated Agentic Search (Tavily + Exa){trace_badge}*"
    )

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
                    label="🟢 Supporter Model (Groq Only)",
                    scale=1
                )
                gemini_model_dropdown = gr.Dropdown(
                    choices=["gemini-2.5-flash", "gemini-2.5-pro"],
                    value="gemini-2.5-flash",
                    label="🔴 Contender Model (Gemini Only)",
                    scale=1
                )

        with gr.Row():
            topic_box = gr.Textbox(
                label="Debate Proposition / Topic (Universal: Any Academic, Technical, Policy, Scientific, or Philosophical Domain)",
                placeholder="Enter any proposition to debate...",
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
            trace_export_btn = gr.DownloadButton(
                "🔍 Download Trace Log (.json)",
                variant="secondary",
                size="lg",
                visible=False,
                scale=1
            )

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

    next_turn_btn.click(
        fn=run_debate_step,
        inputs=[topic_box, loops_slider, cooldown_slider, groq_model_dropdown, gemini_model_dropdown, debate_state],
        outputs=[debate_state, feed_html, next_turn_btn, topic_box, loops_slider, export_btn, trace_export_btn]
    )

    export_btn.click(
        fn=export_transcript_file,
        inputs=[debate_state],
        outputs=[export_btn]
    )

    trace_export_btn.click(
        fn=export_trace_json_file,
        inputs=[debate_state],
        outputs=[trace_export_btn]
    )

    reset_btn.click(
        fn=reset_debate,
        inputs=[],
        outputs=[debate_state, feed_html, next_turn_btn, topic_box, loops_slider, export_btn, trace_export_btn]
    )

if __name__ == "__main__":
    demo.launch()