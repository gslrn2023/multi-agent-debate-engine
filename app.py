# ==============================================================================
# MULTI-AGENT AI DEBATE ENGINE
# Version: v2.8.1 | Date: 2026-11-01
# Changelog (v2.8.1):
#   - Scrubbed all topic-specific examples and domain-biased stopwords to ensure
#     100% universal, domain-agnostic execution across any user proposition.
#   - Implemented 3-line plain-text Agentic Query Planner (DOMAIN / TAVILY / EXA)
#     with thinking_budget=0 on Gemini fallback and epistemic Round 1/2/3 rotation.
#   - Restored "- **[{title} | {url}]:**" snippet headers for accurate citations.
#   - Expanded Tavily & Exa candidate pools to 7 results with a non-empty
#     fallback pass to prevent empty search arrays in later rounds.
#   - Upgraded clean_agent_output to strip leaked section headers and enforce
#     blank lines around bullet lists for proper PDF HTML list rendering.
# ==============================================================================

import os
import re
import json
import time
import tempfile
import threading
import unicodedata
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone
import markdown
import gradio as gr
from groq import Groq
from google import genai
from google.genai import types
from tavily import TavilyClient
from exa_py import Exa
from xhtml2pdf import pisa

# Optional local .env loader for GitHub / local SDK deployments
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==========================================
# PORTABLE ZEROGPU / MULTI-SDK HANDLER
# ==========================================
try:
    import spaces
    HAS_ZEROGPU = True
except ImportError:
    spaces = None
    HAS_ZEROGPU = False

def conditional_zerogpu(duration=5):
    """Applies @spaces.GPU decorator on Hugging Face ZeroGPU; acts as a no-op on CPU/local/GitHub SDKs."""
    def decorator(fn):
        if HAS_ZEROGPU and spaces is not None:
            return spaces.GPU(duration=duration)(fn)
        return fn
    return decorator

@conditional_zerogpu(duration=5)
def zerogpu_healthcheck():
    """Satisfies Hugging Face ZeroGPU mandatory @spaces.GPU startup detection when present."""
    return True

# ==========================================
# VERSION & CONFIGURABLE ENGINE VARIABLES
# ==========================================
APP_VERSION = "v2.8.1"
TRACE = True                # // Set True to enable JSON trace logging & UI download button; Set False for production //
DEFAULT_MAX_LOOPS = 3       # // Default & Max number of debate loops/rounds //
DEFAULT_WAIT_TIME_SEC = 30  # // Wait time in seconds between EACH AI agent execution //
MAX_DAILY_DEBATES = 5       # // Global max debates per day across all users //
DAILY_LIMIT_FILE = "daily_limit.json"
DAILY_LIMIT_LOCK = threading.Lock()
IST_TZ = timezone(timedelta(hours=5, minutes=30))

# Universal list of low-signal social/video/forum domains excluded from web search
EXCLUDED_SEARCH_DOMAINS = [
    "instagram.com",
    "threads.net",
    "threads.com",
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "tiktok.com",
    "reddit.com",
    "x.com",
    "twitter.com",
    "linkedin.com",
    "in.linkedin.com",
    "wallstreetoasis.com",
    "quora.com"
]

# Purely epistemic debate meta-words stripped from Tavily lexical queries to prevent off-topic noise
BANNED_TAVILY_META_WORDS = {
    "empirical", "criticisms", "structural", "limitations", "bottlenecks",
    "discrepancy", "trade-offs", "tradeoffs", "unintended", "consequences",
    "frameworks", "framework", "benchmarks", "validating", "proven",
    "counter-evidence", "counterarguments", "overarching", "comparative",
    "implications", "vulnerabilities", "systemic", "real-world"
}

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
    line-height: 1.58;
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
        raise gr.Error("Missing 'CHK_PSSWD' secret! Please set CHK_PSSWD in your Space Settings or .env file.")

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
# UNIVERSAL AGENTIC QUERY PLANNER & SEARCH AGGREGATION
# ==========================================
def _normalize_url(url: str) -> str:
    """Normalizes URLs (stripping query params except core document IDs) to prevent duplicate articles across rounds."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        id_match = re.search(r'(?:id|Id|PRID)=([0-9A-Za-z_\-]+)', parsed.query)
        id_suffix = f"?id={id_match.group(1).lower()}" if id_match else ""
        return f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}{id_suffix}"
    except Exception:
        return url.strip().lower()

def _normalize_title(title: str) -> str:
    """Normalizes article/study titles so the same study hosted on both PDF and HTML URLs is never duplicated."""
    if not title:
        return ""
    core_title = re.split(r'\s+[\|\-–—]\s+', title.strip())[0]
    alnum = re.sub(r'[^a-z0-9]+', ' ', core_title.lower()).strip()
    words = alnum.split()
    return " ".join(words[:7]) if len(words) >= 3 else alnum

def _compress_topic_anchor(topic: str, max_words: int = 8) -> str:
    """Condenses a long conversational proposition into its core subject nouns using strictly domain-neutral stopwords."""
    neutral_stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "will",
        "have", "has", "had", "not", "but", "can", "more", "than", "into", "their", "its",
        "whether", "should", "would", "could", "really", "actually", "overall", "is", "in",
        "of", "by", "an", "as", "per", "to", "on", "at", "be", "or", "if", "it", "many"
    }
    tokens = re.findall(r'\b[A-Za-z0-9\.\-%]+\b', topic)
    seen_tok = set()
    core = []
    for t in tokens:
        lt = t.lower()
        if lt not in neutral_stopwords and lt not in seen_tok:
            seen_tok.add(lt)
            core.append(t)
    selected = core[:max_words] if core else tokens[:max_words]
    return " ".join(selected)

def _clean_snippet_boilerplate(text: str) -> str:
    """Strips embargo notices, repeated titles, and navigation boilerplate from search snippets."""
    if not text:
        return ""
    cleaned = re.sub(r'\s+', ' ', text).strip()
    cleaned = re.sub(
        r'This Press Release is embargoed against publication[^\.]{0,120}(?:\.|\b202[0-9]\b)',
        '',
        cleaned,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r'^(?:\|\s*)+', '', cleaned).strip()
    return cleaned

def _sanitize_tavily_query(raw_query: str, fallback_anchor: str) -> str:
    """Strips banned abstract debate meta-words from Tavily query so only concrete domain nouns and metrics remain."""
    if not raw_query:
        return fallback_anchor
    tokens = re.findall(r'\b[A-Za-z0-9\.\-%]+\b', raw_query)
    filtered = [t for t in tokens if t.lower() not in BANNED_TAVILY_META_WORDS]
    result = " ".join(filtered[:11]).strip()
    return result if len(result) >= 8 else fallback_anchor

def extract_opponent_focus_terms(last_turn_text: str) -> str:
    """
    Universal Per-Bullet Concept Extractor:
    Extracts concrete subject nouns and official dataset/institutional acronyms from each bullet body
    while filtering out generic debate verbs and media outlet names.
    """
    if not last_turn_text:
        return ""

    bullet_lines = [
        line.strip() for line in last_turn_text.splitlines()
        if re.match(r'^[\*\-\•]\s+', line.strip())
    ]
    if not bullet_lines:
        bullet_lines = [last_turn_text]

    generic_and_media_stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "while", "are", "was", "were",
        "have", "has", "had", "not", "but", "what", "all", "can", "more", "than", "into",
        "over", "under", "their", "they", "its", "also", "such", "only", "which", "when",
        "round", "supporter", "contender", "however", "because", "between", "through",
        "argument", "claim", "claims", "evidence", "study", "studies", "report", "reports",
        "shows", "prove", "proves", "point", "points", "reality", "impact", "impacts",
        "direct", "rebuttal", "front", "independent", "first", "second", "third", "every",
        "methodological", "opacity", "calculation", "divergence", "vital", "metrics",
        "broad", "based", "sectoral", "surge", "momentum", "highlights", "notes",
        "suggests", "indicates", "indicating", "challenges", "undermined", "despite",
        "significant", "underlying", "headline", "figure", "figures", "print", "published",
        "cannot", "verify", "across", "sectors", "strongest", "quarterly", "cites",
        "dismisses", "confirms", "leans", "asserts", "reveals", "according", "official",
        "statistics", "former", "interview", "commentary", "breakingviews", "reuters",
        "ndtv", "cnbc", "livemint", "mint", "express", "hindu", "times", "bloomberg",
        "bbc", "journal", "analysis", "research", "merely", "citing", "higher", "lower",
        "emphasizes", "assertion", "recent", "integrated", "indexed", "multilevel",
        "analyses", "multivariate", "finds", "nationally", "representative", "provided",
        "investigates", "almost", "strongly", "where", "regions", "live", "web"
    }

    seen = set()
    extracted = []

    for b_line in bullet_lines:
        body_only = re.sub(r'^\s*[\*\-\•]\s*(\*\*[^\*]{1,75}\*\*[:\-–—]?)?', ' ', b_line)
        acronyms = re.findall(r'\b[A-Z]{3,6}\b', body_only)
        words = re.findall(r'\b[A-Za-z][a-z\-]{4,17}\b', body_only)

        picked_this_bullet = 0
        for token in (acronyms + words):
            lw = token.lower().strip("-")
            if lw not in generic_and_media_stopwords and lw not in seen and len(lw) >= 3:
                seen.add(lw)
                extracted.append(token if token.isupper() else lw)
                picked_this_bullet += 1
                if picked_this_bullet >= 2:
                    break
        if len(extracted) >= 8:
            break

    return " ".join(extracted)

def _parse_three_line_plan(raw_text: str, short_anchor: str) -> dict | None:
    """Parses the 3-line DOMAIN / TAVILY / EXA plain-text output from the query planner."""
    if not raw_text:
        return None
    dom_match = re.search(r'DOMAIN\s*[:\-]\s*(.+)', raw_text, re.IGNORECASE)
    tav_match = re.search(r'TAVILY\s*[:\-]\s*(.+)', raw_text, re.IGNORECASE)
    exa_match = re.search(r'EXA\s*[:\-]\s*(.+)', raw_text, re.IGNORECASE)

    if not tav_match or not exa_match:
        return None

    inferred_domain = dom_match.group(1).strip().strip('"\'*') if dom_match else "Specialized Domain"
    raw_tav = tav_match.group(1).strip().strip('"\'*')
    raw_exa = exa_match.group(1).strip().strip('"\'*')

    clean_tav = _sanitize_tavily_query(raw_tav, short_anchor)
    clean_exa = raw_exa[:350].strip()

    if len(clean_tav) >= 8 and len(clean_exa) >= 15:
        return {
            "inferred_domain": inferred_domain[:50],
            "tavily_query": clean_tav,
            "exa_query": clean_exa
        }
    return None

def generate_agentic_search_queries(
    topic: str,
    agent_role: str,
    current_round: int,
    last_opponent_text: str = ""
) -> dict:
    """
    Universal Agentic Query Planner (v2.8.1):
    Dynamically infers the subject domain of any user proposition and generates domain-native
    search queries using a resilient 3-line plain-text protocol (DOMAIN / TAVILY / EXA).
    """
    short_anchor = _compress_topic_anchor(topic, max_words=8)
    opp_terms = extract_opponent_focus_terms(last_opponent_text)

    stance_desc = (
        f"SUPPORTER proving that '{topic}' holds true and delivers net positive outcomes"
        if agent_role == "Supporter"
        else f"CONTENDER proving that '{topic}' is flawed, overstated, or counterproductive"
    )

    # Purely epistemic, domain-agnostic round rotation to ensure fresh evidence dimensions each round
    if current_round == 1:
        round_angle = (
            "ROUND 1 FOCUS (Foundational Baselines & Primary Datasets): Search for headline quantitative rates, "
            "flagship national/global datasets, primary empirical benchmarks, and core institutional studies."
        )
    elif current_round == 2:
        round_angle = (
            "ROUND 2 FOCUS (Methodological Audits, Confounding Variables & Secondary Indicators): Search for "
            "secondary real-world indicators, sample/measurement biases, hidden trade-offs, or operational "
            "data that directly counter the opponent's previous claims."
        )
    else:
        round_angle = (
            "ROUND 3 FOCUS (Longitudinal Outcomes & Structural Comparisons): Search for long-term longitudinal "
            "trajectories, cross-regional or cross-demographic comparisons, systemic constraints, and enduring precedents."
        )

    opp_context = (
        f"Opponent's previous claims to refute:\n{last_opponent_text[:500]}\n"
        if last_opponent_text
        else "Opening turn of the debate.\n"
    )

    planner_prompt = (
        f"You are an expert search query strategist for a universal, multi-domain debate engine.\n"
        f"Proposition: \"{topic}\"\n"
        f"Assigned Role: {stance_desc}\n"
        f"{round_angle}\n"
        f"{opp_context}\n"
        f"Task: Identify the exact academic/professional domain of this proposition and formulate search queries "
        f"using concrete, domain-native nouns, specific measurable indicators, and relevant dataset or institution acronyms.\n"
        f"- NEVER use abstract debate filler words in TAVILY like 'empirical', 'criticisms', 'structural', 'limitations', 'bottlenecks', 'frameworks', or 'consequences'.\n"
        f"- Use only concrete subject terms, specific metrics, and official data sources native to the proposition's field.\n\n"
        f"Reply in EXACTLY 3 plain-text lines (no JSON, no markdown, no extra commentary):\n"
        f"DOMAIN: <2 to 4 word academic or industry domain name>\n"
        f"TAVILY: <7 to 10 concrete domain keywords, specific metric names, and dataset/institution acronyms>\n"
        f"EXA: <14 to 22 word natural-language scholarly query seeking specific empirical findings and studies for this round's focus>"
    )

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    planner_errors = []

    # Attempt 1: Gemini 2.5 Flash with thinking_budget=0 (fast, zero-thinking, high instruction adherence)
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=planner_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=250,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            parsed = _parse_three_line_plan(resp.text or "", short_anchor)
            if parsed:
                return {
                    **parsed,
                    "planner_source": "gemini:gemini-2.5-flash",
                    "extracted_opponent_keywords": opp_terms,
                    "planner_errors": planner_errors
                }
            else:
                planner_errors.append(f"Gemini format mismatch: {(resp.text or '')[:100]}")
        except Exception as e:
            planner_errors.append(f"Gemini error: {str(e)[:120]}")

    # Attempt 2: Groq llama-3.3-70b-versatile / llama-3.1-8b-instant plain-text call
    if groq_key:
        for g_model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            try:
                client = Groq(api_key=groq_key)
                resp = client.chat.completions.create(
                    model=g_model,
                    messages=[{"role": "user", "content": planner_prompt}],
                    temperature=0.2,
                    max_completion_tokens=200
                )
                raw_out = resp.choices[0].message.content or ""
                parsed = _parse_three_line_plan(raw_out, short_anchor)
                if parsed:
                    return {
                        **parsed,
                        "planner_source": f"groq:{g_model}",
                        "extracted_opponent_keywords": opp_terms,
                        "planner_errors": planner_errors
                    }
                else:
                    planner_errors.append(f"{g_model} format mismatch: {raw_out[:100]}")
            except Exception as e:
                planner_errors.append(f"{g_model} error: {str(e)[:120]}")

    # Attempt 3: Epistemic round-varied deterministic fallback (100% domain-neutral)
    round_suffixes = {
        1: "statistics rate percentage survey dataset study",
        2: "methodology bias confounding factors secondary outcomes data",
        3: "longitudinal study long-term trends comparative analysis"
    }
    r_suf = round_suffixes.get(current_round, "statistics survey data")
    fallback_tavily = _sanitize_tavily_query(f"{short_anchor} {opp_terms} {r_suf}", short_anchor)
    fallback_exa = f"Peer-reviewed study and statistical findings on {short_anchor} {opp_terms} ({r_suf})"[:350]

    return {
        "inferred_domain": "Universal Fallback",
        "planner_source": "deterministic_fallback",
        "tavily_query": fallback_tavily,
        "exa_query": fallback_exa,
        "extracted_opponent_keywords": opp_terms,
        "planner_errors": planner_errors
    }

def fetch_combined_research(
    topic: str,
    agent_role: str,
    current_round: int,
    last_opponent_text: str = "",
    seen_urls: list = None,
    seen_titles: list = None
) -> tuple[str, dict, list, list]:
    """
    Executes Agentic Query Planning followed by deduplicated Tavily + Exa search (with pool=7 and non-empty fallback).
    Returns: (formatted_snippets_string, trace_metadata_dict, updated_seen_urls, updated_seen_titles)
    """
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    exa_key = os.environ.get("EXA_API_KEY", "").strip()

    if not tavily_key and not exa_key:
        raise gr.Error("Missing both TAVILY_API_KEY and EXA_API_KEY in Space Secrets.")

    seen_url_set = set(seen_urls or [])
    seen_title_set = set(seen_titles or [])

    # 1. Agentic Domain Query Formulation
    plan = generate_agentic_search_queries(topic, agent_role, current_round, last_opponent_text)
    t_query = plan["tavily_query"]
    e_query = plan["exa_query"]
    short_anchor = _compress_topic_anchor(topic, max_words=8)

    snippets = []
    tavily_trace_records = []
    exa_trace_records = []

    # 2. TAVILY SEARCH (Pool of 7 candidates, full [Title | URL] headers)
    if tavily_key:
        try:
            tavily_client = TavilyClient(api_key=tavily_key)
            t_resp = tavily_client.search(
                query=t_query,
                search_depth="basic",
                max_results=7,
                include_answer=(current_round == 1),
                exclude_domains=EXCLUDED_SEARCH_DOMAINS
            )
            if current_round == 1 and t_resp.get("answer"):
                ans_text = _clean_snippet_boilerplate(t_resp["answer"])[:360]
                if len(ans_text) >= 80 and "do not provide direct information" not in ans_text.lower():
                    snippets.append(f"- **[Web Synthesis]:** {ans_text}")
                    tavily_trace_records.append({
                        "type": "tavily_answer",
                        "title": "Web Synthesis",
                        "url": "N/A",
                        "snippet": ans_text
                    })

            added_tavily = 0
            fallback_tavily_candidates = []
            for res in t_resp.get("results", []):
                url = res.get("url", "")
                title = res.get("title", "Verified Report").strip()
                norm_u = _normalize_url(url)
                norm_t = _normalize_title(title)

                if any(dom in norm_u for dom in EXCLUDED_SEARCH_DOMAINS):
                    continue

                raw_content = res.get("content", "")
                content = _clean_snippet_boilerplate(raw_content)[:440]
                if len(content) < 110:
                    continue

                if norm_u in seen_url_set or (norm_t and norm_t in seen_title_set):
                    fallback_tavily_candidates.append((title, url, content, norm_u, norm_t))
                    continue

                seen_url_set.add(norm_u)
                if norm_t:
                    seen_title_set.add(norm_t)
                snippets.append(f"- **[{title} | {url}]:** {content}...")
                tavily_trace_records.append({
                    "type": "tavily_result",
                    "title": title,
                    "url": url,
                    "snippet": content
                })
                added_tavily += 1
                if added_tavily >= 2:
                    break

            # Non-empty fallback if strict deduplication filtered all candidates
            if added_tavily == 0 and fallback_tavily_candidates:
                title, url, content, norm_u, norm_t = fallback_tavily_candidates[0]
                snippets.append(f"- **[{title} | {url}]:** {content}...")
                tavily_trace_records.append({
                    "type": "tavily_fallback_result",
                    "title": title,
                    "url": url,
                    "snippet": content
                })
        except Exception as e:
            err_str = str(e)
            gr.Warning(f"⚠️ Tavily search warning: {err_str}. Continuing with Exa/internal data.")
            tavily_trace_records.append({"error": err_str})

    # 3. EXA NEURAL SEARCH (Pool of 7 candidates, full [Title | URL] headers)
    if exa_key:
        try:
            exa_client = Exa(api_key=exa_key)
            e_resp = exa_client.search_and_contents(
                e_query,
                type="auto",
                num_results=7,
                exclude_domains=EXCLUDED_SEARCH_DOMAINS,
                highlights={
                    "num_sentences": 3,
                    "highlights_per_url": 1,
                    "query": f"{t_query} statistical findings percentages empirical results conclusion"
                },
                text={"max_characters": 550}
            )
            added_exa = 0
            fallback_exa_candidates = []
            for res in getattr(e_resp, "results", []):
                url = getattr(res, "url", "")
                title = (getattr(res, "title", None) or "Academic Study").strip()
                norm_u = _normalize_url(url)
                norm_t = _normalize_title(title)

                if any(dom in norm_u for dom in EXCLUDED_SEARCH_DOMAINS):
                    continue

                highlights_list = getattr(res, "highlights", None) or []
                raw_text = " ".join(highlights_list) if highlights_list else (getattr(res, "text", "") or "")
                text_body = _clean_snippet_boilerplate(raw_text)[:460]
                if len(text_body) < 100:
                    continue

                if norm_u in seen_url_set or (norm_t and norm_t in seen_title_set):
                    fallback_exa_candidates.append((title, url, text_body, norm_u, norm_t))
                    continue

                seen_url_set.add(norm_u)
                if norm_t:
                    seen_title_set.add(norm_t)
                snippets.append(f"- **[{title} | {url}]:** {text_body}...")
                exa_trace_records.append({
                    "type": "exa_result",
                    "title": title,
                    "url": url,
                    "snippet": text_body
                })
                added_exa += 1
                if added_exa >= 2:
                    break

            # Non-empty fallback if strict deduplication filtered all candidates
            if added_exa == 0 and fallback_exa_candidates:
                title, url, text_body, norm_u, norm_t = fallback_exa_candidates[0]
                snippets.append(f"- **[{title} | {url}]:** {text_body}...")
                exa_trace_records.append({
                    "type": "exa_fallback_result",
                    "title": title,
                    "url": url,
                    "snippet": text_body
                })
        except Exception as e:
            err_str = str(e)
            gr.Warning(f"⚠️ Exa search warning: {err_str}. Continuing with Tavily/internal data.")
            exa_trace_records.append({"error": err_str})

    if not snippets:
        combined_text = "No external web snippets returned; rely on verified historical, scientific, and empirical facts."
    else:
        combined_text = "\n".join(snippets)

    trace_meta = {
        "inferred_domain": plan["inferred_domain"],
        "query_planner_source": plan["planner_source"],
        "planner_errors": plan.get("planner_errors", []),
        "compressed_topic_anchor": short_anchor,
        "extracted_opponent_keywords": plan["extracted_opponent_keywords"],
        "tavily_dispatched_query": t_query,
        "exa_dispatched_prompt": e_query,
        "tavily_results": tavily_trace_records,
        "exa_results": exa_trace_records,
        "combined_research_char_length": len(combined_text),
        "total_unique_urls_seen_so_far": len(seen_url_set),
        "total_unique_titles_seen_so_far": len(seen_title_set)
    }

    return combined_text, trace_meta, list(seen_url_set), list(seen_title_set)

# ==========================================
# OUTPUT SANITIZER (STRIPS LEAKED LABELS & FIXES MARKDOWN LISTS)
# ==========================================
def clean_agent_output(text: str) -> str:
    """
    1. Strips standalone section headers ('Opening Hook', 'Fact-Sheet', 'Closing Takeaway').
    2. Strips mechanical bullet prefixes ('Direct Rebuttal:', 'Methodological audit -').
    3. Ensures a blank line before the bullet list so markdown.markdown() renders proper HTML <ul><li> lists.
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

# ==========================================
# PHASE 2: EXTERNAL PROMPT LOADER & API CALLS
# ==========================================
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

def call_groq_supporter(system_prompt: str, user_prompt: str, groq_model: str, temperature: float = 0.35) -> str:
    """Calls Groq API with calibrated temperature (0.35) to enforce source-bound precision."""
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

    return call_with_rate_limit_retry(_execute, "Groq")

def call_gemini_contender(system_prompt: str, user_prompt: str, gemini_model: str, temperature: float = 0.35) -> str:
    """Calls Gemini API with calibrated temperature (0.35) and dedicated thinking budget."""
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

# ==========================================
# PDF TEXT SANITIZER, PDF EXPORTER & JSON TRACE EXPORTER
# ==========================================
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

# ==========================================
# MAIN STEP-BY-STEP DEBATE ORCHESTRATOR
# ==========================================
def run_debate_step(topic_input, max_loops_input, cooldown_input, groq_model, gemini_model, state):
    """Executes a single step-by-step turn with Agentic Query Planning, deduplicated search, PDF export, and TRACE JSON logging."""
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
    is_very_first_speech = (turn_idx == 0)

    # Enforce wait time before EVERY agent execution (after the very first opening turn)
    wait_sec = int(cooldown_input)
    enforce_wait_time(state.get("last_agent_ts", 0.0), wait_sec, f"{agent_role} (Round {current_round})")

    # Extract opponent's last speech text
    last_opponent_text = state["history"][-1]["content"] if state["history"] else ""

    # Fetch deduplicated facts & figures via Agentic Query Planner + Tavily + Exa
    web_facts, trace_meta, updated_seen_urls, updated_seen_titles = fetch_combined_research(
        topic=current_topic,
        agent_role=agent_role,
        current_round=current_round,
        last_opponent_text=last_opponent_text,
        seen_urls=state.get("seen_urls", []),
        seen_titles=state.get("seen_titles", [])
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
        f"2. CITE EXACT SOURCE TITLES FROM RESEARCH: Each research snippet above starts with **[Title | URL]:**. Cite the actual study, dataset, journal, or institution name in parentheses—e.g., (Institution or Journal Name, Year). NEVER write '(Live Web Research)' or invent generic placeholder study names.\n"
        f"3. ZERO NUMERICAL FABRICATION OR DIRECTIONAL INVERSION: Never invent percentages, sample sizes, or metrics not explicitly present in the research snippets. Never invert the directional finding of a source (if a snippet reports a metric that favors the opposing view, do not claim the snippet favors your view—instead, audit its methodology, baseline scope, or confounding variables).\n"
        f"4. IGNORE OFF-TOPIC SNIPPETS: If any search snippet is unrelated to the proposition, ignore it completely."
    )

    if not state["history"]:
        user_prompt = f"{research_block}\n\nDeliver your Round 1 opening argument on: '{current_topic}'."
    else:
        transcript = build_tpm_safe_transcript(state["history"])
        user_prompt = (
            f"Debate transcript so far:\n\n{transcript}\n\n"
            f"{research_block}\n\n"
            f"Deliver your Round {current_round} response as the {agent_role} "
            f"(4 to 5 bullets: 2-3 direct rebuttals + 1-2 fresh evidence fronts, 180-230 words total, with zero section headers)."
        )

    if agent_role == "Supporter":
        raw_reply = call_groq_supporter(system_prompt, user_prompt, groq_model, temperature=0.35)
        used_model = groq_model
    else:
        raw_reply = call_gemini_contender(system_prompt, user_prompt, gemini_model, temperature=0.35)
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

    # Record structured trace entry when TRACE is ON
    if TRACE:
        turn_trace_entry = {
            "turn_number": turn_idx + 1,
            "round": current_round,
            "role": agent_role,
            "model": used_model,
            "timestamp_ist": datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S IST'),
            "inferred_domain": trace_meta["inferred_domain"],
            "query_planner_source": trace_meta["query_planner_source"],
            "planner_errors": trace_meta.get("planner_errors", []),
            "compressed_topic_anchor": trace_meta["compressed_topic_anchor"],
            "extracted_opponent_keywords": trace_meta["extracted_opponent_keywords"],
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
        print(f"\n{'='*60}\n[TRACE {APP_VERSION}] Round {current_round} | {agent_role} ({used_model})")
        print(f"  -> Inferred Domain   : {trace_meta['inferred_domain']} ({trace_meta['query_planner_source']})")
        if trace_meta.get("planner_errors"):
            print(f"  -> Planner Warnings  : {trace_meta['planner_errors']}")
        print(f"  -> Tavily Query      : {trace_meta['tavily_dispatched_query']}")
        print(f"  -> Exa Prompt        : {trace_meta['exa_dispatched_prompt']}")
        print(f"  -> Snippet Payload   : {trace_meta['combined_research_char_length']} chars | Unique URLs: {trace_meta['total_unique_urls_seen_so_far']}\n{'='*60}")
    else:
        print(f"\n{'='*50}\n[DEBATE TURN COMPLETED] Topic: {state['topic']} | Round {current_round} | {agent_role} ({used_model})\n{'='*50}")

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

    # Prepare updated PDF file and optional Trace JSON file
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
# PHASE 3: GRADIO INTERFACE (WITH PASSWORD GATE)
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
        f"*🟢 Supporter (Groq `openai/gpt-oss-120b`) vs. 🔴 Contender (Google `gemini-2.5-flash`) • Universal Agentic Search (Tavily + Exa){trace_badge}*"
    )

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