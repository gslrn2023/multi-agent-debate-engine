---
title: Multi-Agent AI Debate Engine
emoji: ⚖️
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# ⚖️ Multi-Agent AI Debate Engine (`v2.9.6`)

A universal, domain-agnostic **Multi-Agent AI Debate Engine** featuring strict dual-provider isolation, autonomous **5-Line Cognitive Search Planning**, live lexical and neural web retrieval (**Tavily + Exa**), anti-hallucination guardrails, paginated PDF transcript generation, and structured JSON forensic trace logging.

---

## 🌟 Key Architectural Highlights

1. **Strict Dual-Provider Isolation (Zero Shared Rate Limits):**
   - **🟢 Agent A (The Supporter):** Powered exclusively by **Groq** (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`, or `llama-3.1-8b-instant`) for both cognitive query planning and speech generation.
   - **🔴 Agent B (The Contender):** Powered exclusively by **Google Gemini** (`gemini-2.5-flash` or `gemini-2.5-pro`) for both cognitive query planning and speech generation.
2. **5-Line Cognitive Research Planner (`search_engine.py`):**
   Before every turn, each agent autonomously executes a 5-step reasoning chain with zero domain hardcoding:
   - `DOMAIN`: Identifies the exact academic, scientific, legal, technical, or policy field.
   - `CLAUSES_AND_TARGET`: Decomposes the proposition into distinct claims and performs **Proposition-vs-Speech Gap Analysis** (targeting both direct rebuttals and omitted/unaddressed clauses).
   - `AUTHORITIES`: Identifies gold-standard statistical agencies, regulatory bodies, or benchmark datasets native to that domain.
   - `TAVILY`: Dispatches a precision lexical query for institutional reports and statistical tables.
   - `EXA`: Dispatches a neural semantic query for peer-reviewed studies and empirical evaluations.
3. **Epistemic Integrity & Anti-Hallucination Guardrails:**
   - **Qualitative Parity:** Agents may only cite numbers or percentages explicitly present in the retrieved snippets; qualitative findings are argued qualitatively without fabricating metrics.
   - **Strict Source Ownership:** Agents must cite sources strictly from their own turn's retrieved research block and are forbidden from hijacking opponent citations or inverting hostile study conclusions.
   - **Cross-Round Deduplication:** URLs, study titles, and Tavily queries are deduplicated across all rounds.
4. **Cumulative Final-Round Verdict:**
   In the final round, both agents synthesize their strongest verified evidence from Round 1 through the Final Round into an overarching closing conclusion.
5. **Paginated PDF & Forensic JSON Trace Exports:**
   - Download a formatted, paginated **`.pdf` Debate Transcript** after any turn.
   - When `TRACE=true`, download a complete **`{topic}_trace_log.json`** containing every inferred domain, clause target, authority list, search query, raw snippet, and model output.

---

## 🔑 Required Secrets & Environment Variables

Whether running locally via `.env`, on **Hugging Face Spaces**, or on **Kaggle**, configure the following 5 secrets:

| Variable Name | Required | Description | Where to Get It |
| :--- | :---: | :--- | :--- |
| `CHK_PSSWD` | **Yes** | Password required to unlock the web UI and protect your API quotas. | Choose any strong password (e.g., `my_secret_pass`) |
| `GROQ_API_KEY` | **Yes** | API key for Agent A (Supporter) query planner & speech generation. | [Groq Console](https://console.groq.com/keys) |
| `GEMINI_API_KEY` | **Yes** | API key for Agent B (Contender) query planner & speech generation. | [Google AI Studio](https://aistudio.google.com/apikey) |
| `TAVILY_API_KEY` | **Yes** | API key for lexical web & institutional report search. | [Tavily Search API](https://tavily.com/) |
| `EXA_API_KEY` | **Yes** | API key for neural semantic & academic paper search. | [Exa AI Dashboard](https://dashboard.exa.ai/) |

---

## 🚀 Installation & Deployment Guide

### Path A: Run on a Local Machine (Laptop / Desktop)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
   cd <your-repo-name>