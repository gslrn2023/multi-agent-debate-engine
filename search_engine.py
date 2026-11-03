# ==============================================================================
# MODULE: search_engine.py
# Version: v2.9.3 | Date: 2026-11-03
# Purpose: Strict single-API per-agent Query Planner (Supporter -> Groq ONLY,
#          Contender -> Gemini ONLY), stance-polarized query formulation,
#          and deduplicated Tavily + Exa retrieval.
# ==============================================================================

import os
import re
import time
from urllib.parse import urlparse
import gradio as gr
from groq import Groq
from google import genai
from google.genai import types
from tavily import TavilyClient
from exa_py import Exa

from config import (
    EXCLUDED_SEARCH_DOMAINS,
    BANNED_TAVILY_META_WORDS
)
from llm_agents import call_with_rate_limit_retry

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

def _extract_clean_domain_label(url: str) -> str:
    """Extracts a clean institution/publisher domain label from a URL for citation clarity."""
    if not url:
        return "Verified Source"
    try:
        host = urlparse(url.strip()).netloc.lower()
        host = re.sub(r'^(?:www\.|in\.|m\.)', '', host)
        return host if host else "Verified Source"
    except Exception:
        return "Verified Source"

def _normalize_title(title: str) -> str:
    """Normalizes article/study titles so the same study hosted on both PDF and HTML URLs is never duplicated."""
    if not title:
        return ""
    core_title = re.split(r'\s+[\|\-–—]\s+', title.strip())[0]
    alnum = re.sub(r'[^a-z0-9]+', ' ', core_title.lower()).strip()
    words = alnum.split()
    return " ".join(words[:7]) if len(words) >= 3 else alnum

def _clean_display_title(title: str, url: str) -> str:
    """
    Cleans trailing ellipses ('...'), site suffixes, and generic/corrupted PDF metadata
    (e.g. 'Combined PDF', 'TX_1~ABS:AT'), rescuing human-readable filenames from PDF URLs when needed.
    """
    domain_label = _extract_clean_domain_label(url)
    if not title:
        return f"Study via {domain_label}"

    cleaned = re.sub(r'\s*\.\.\.+\s*$', '', title.strip())
    cleaned = re.sub(r'^\s*\(PDF\)\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r'\s+[\|\-–—]\s+(?:Get\s+Access|PDF|Full\s+Text|ScienceDirect|ResearchGate|PubMed|PMC).*$',
        '',
        cleaned,
        flags=re.IGNORECASE
    )

    is_corrupted = bool(re.search(r'[~_]{1,}|ABS:|TX_\d', cleaned))
    generic_titles = {
        "academic study", "verified report", "chapter 1 introduction",
        "research analysis", "abstract", "introduction", "home",
        "combined pdf", "full text pdf", "untitled", "document"
    }

    if is_corrupted or len(cleaned) < 8 or cleaned.lower() in generic_titles:
        # 1. Try rescuing filename from query parameter
        fn_match = re.search(r'filename%3D%22([^%]+)', url)
        if fn_match:
            rescued = fn_match.group(1).replace('+', ' ').strip()
            if len(rescued) >= 8:
                return rescued
        # 2. Try rescuing filename from direct .pdf URL path
        try:
            path_base = os.path.basename(urlparse(url.strip()).path)
            if path_base.lower().endswith(".pdf"):
                stem = re.sub(r'\.pdf$', '', path_base, flags=re.IGNORECASE)
                stem = re.sub(r'[_\-]+', ' ', stem).strip()
                if len(stem) >= 10 and not stem.isdigit():
                    return f"{stem} ({domain_label})"
        except Exception:
            pass
        return f"Research Study ({domain_label})"

    return cleaned

def _compress_topic_anchor(topic: str, max_words: int = 8) -> str:
    """Condenses a long conversational proposition into its core subject nouns using domain-neutral stopwords."""
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

def extract_opponent_focus_terms(last_turn_text: str, topic: str = "") -> str:
    """
    Universal Per-Bullet Concept Extractor:
    Extracts concrete subject nouns and official dataset acronyms from each bullet body,
    filtering out citation wrapper words (e.g., 'COVID' if not in topic) and generic debate verbs.
    """
    if not last_turn_text:
        return ""

    topic_lower = (topic or "").lower()

    # Strip parenthetical citations at the end of bullets first so container titles (like '(... during COVID, 2022)')
    # never pollute the extracted search keywords
    text_without_citations = re.sub(r'\([^\)\n]{3,80}\)', ' ', last_turn_text)

    bullet_lines = [
        line.strip() for line in text_without_citations.splitlines()
        if re.match(r'^[\*\-\•]\s+', line.strip())
    ]
    if not bullet_lines:
        bullet_lines = [text_without_citations]

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
        "investigates", "almost", "strongly", "where", "regions", "live", "web",
        "opponent", "level", "committee", "attributing", "informal", "inherently"
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
            if lw == "covid" and "covid" not in topic_lower and "pandemic" not in topic_lower:
                continue
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
    active_model: str,
    last_opponent_text: str = "",
    past_agent_queries: list = None
) -> dict:
    """
    Strict Single-Provider Agentic Query Planner (v2.9.3):
      - If agent_role == "Supporter": Calls ONLY Groq (active_model).
      - If agent_role == "Contender": Calls ONLY Gemini (active_model).
    Enforces Core Subject/Jurisdiction Retention, multi-prong rotation, and affirmative
    verification keywords for the Supporter to prevent retrieving hostile articles.
    """
    short_anchor = _compress_topic_anchor(topic, max_words=8)
    opp_terms = extract_opponent_focus_terms(last_opponent_text, topic)
    past_queries_list = past_agent_queries or []

    if agent_role == "Supporter":
        stance_directive = (
            f"You are Agent A (THE SUPPORTER). Your goal is to PROVE that \"{topic}\" holds true across all its stated sub-claims. "
            f"Search specifically for datasets, empirical rates, and institutional studies that validate your proposition.\n"
            f"CRITICAL SUPPORTER QUERY RULE: If the proposition claims the absence of negative flaws, do NOT copy those negative flaw nouns into your TAVILY or EXA queries (as negative keywords pull critical articles). "
            f"Instead, translate them into affirmative verification metrics, compliance rates, audit mechanisms, and positive performance indicators."
        )
    else:
        stance_directive = (
            f"You are Agent B (THE CONTENDER). Your goal is to REFUTE \"{topic}\" by dismantling every specific sub-claim in the proposition. "
            f"Search specifically for counter-statistics, technical/operational failure rates, hidden costs, regulatory loopholes, and empirical flaws in the Supporter's claims."
        )

    if current_round == 1:
        round_angle = (
            "ROUND 1 FOCUS (Primary Rates, Flagship Datasets & Multi-Prong Baselines): Search for headline empirical rates, "
            "flagship institutional datasets, and primary performance metrics covering the core sub-claims of the proposition."
        )
    elif current_round == 2:
        round_angle = (
            "ROUND 2 FOCUS (Unchallenged Prongs, Technical/Operational Trade-offs & Confounding Factors): Scan the opponent's Round 1 speech "
            "and target the specific technical, efficiency, cost, or governance sub-claims that have not yet been audited. Search for direct "
            "counter-indicators, measurement biases, and operational bottlenecks."
        )
    else:
        round_angle = (
            "ROUND 3 FOCUS (Macro-Structural, Regional/Distributional & Long-Term Viability): Pivot to a fresh structural dimension! "
            "Search for supply-chain/infrastructure constraints, demographic/regional disparities, regulatory enforcement gaps, and long-term sustainability data."
        )

    past_memory_block = ""
    if past_queries_list:
        joined_past = " | ".join(past_queries_list)
        past_memory_block = (
            f"QUERIES YOU ALREADY RAN IN EARLIER ROUNDS: [{joined_past}]\n"
            f"CRITICAL ANCHOR & DIVERSITY RULE:\n"
            f"1. ALWAYS RETAIN THE CORE SUBJECT & JURISDICTION: You MUST keep the primary subject noun and country/region/jurisdiction from the proposition in your TAVILY and EXA queries on every round so search results never drift to unrelated countries or industries.\n"
            f"2. ROTATE THE ANALYTICAL METRICS: Do NOT repeat the same secondary aspect or metric keywords from your earlier queries above; pivot to a new sub-claim or technical dimension of the proposition.\n"
        )

    opp_context = (
        f"Opponent's previous claims to dismantle:\n{last_opponent_text[:550]}\n"
        if last_opponent_text
        else "Opening turn of the debate.\n"
    )

    planner_prompt = (
        f"{stance_directive}\n"
        f"{round_angle}\n"
        f"{past_memory_block}"
        f"{opp_context}\n"
        f"Task: Formulate search queries using concrete, domain-native nouns, specific measurable indicators, and official dataset or institution acronyms.\n"
        f"- NEVER use abstract debate filler words in TAVILY like 'empirical', 'criticisms', 'structural', 'limitations', 'bottlenecks', 'frameworks', or 'consequences'.\n"
        f"- Always include the proposition's core subject entity and geographical/institutional scope in TAVILY.\n\n"
        f"Reply in EXACTLY 3 plain-text lines (no JSON, no markdown, no extra text):\n"
        f"DOMAIN: <2 to 4 word academic or industry domain name>\n"
        f"TAVILY: <core subject + jurisdiction/scope + 5 to 8 concrete domain metric keywords and dataset/institution acronyms>\n"
        f"EXA: <14 to 22 word natural-language scholarly query seeking specific empirical findings and studies backing your stance>"
    )

    planner_errors = []

    # ==========================================================
    # STRICT ISOLATION: SUPPORTER USES ONLY GROQ
    # ==========================================================
    if agent_role == "Supporter":
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                def _groq_plan():
                    client = Groq(api_key=groq_key)
                    kwargs = {
                        "model": active_model,
                        "messages": [{"role": "user", "content": planner_prompt}],
                        "temperature": 0.25,
                        "max_completion_tokens": 600 if "gpt-oss" in active_model else 220,
                    }
                    if "gpt-oss" in active_model:
                        kwargs["reasoning_effort"] = "low"
                    resp = client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""

                raw_out = call_with_rate_limit_retry(
                    _groq_plan,
                    "Groq Query Planner",
                    max_retries=2,
                    base_wait=10,
                    raise_ui_error=False
                )
                parsed = _parse_three_line_plan(raw_out, short_anchor)
                if parsed:
                    return {
                        **parsed,
                        "planner_source": f"groq:{active_model}",
                        "extracted_opponent_keywords": opp_terms,
                        "planner_errors": planner_errors
                    }
                else:
                    planner_errors.append(f"Groq planner format mismatch: {raw_out[:120]}")
            except Exception as e:
                planner_errors.append(f"Groq planner error ({active_model}): {str(e)[:140]}")

    # ==========================================================
    # STRICT ISOLATION: CONTENDER USES ONLY GEMINI
    # ==========================================================
    else:
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if gemini_key:
            try:
                def _gemini_plan():
                    client = genai.Client(api_key=gemini_key)
                    cfg_kwargs = {
                        "temperature": 0.25,
                        "max_output_tokens": 300 if "flash" in active_model else 1024,
                    }
                    if "flash" in active_model:
                        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                    resp = client.models.generate_content(
                        model=active_model,
                        contents=planner_prompt,
                        config=types.GenerateContentConfig(**cfg_kwargs)
                    )
                    return resp.text or ""

                raw_out = call_with_rate_limit_retry(
                    _gemini_plan,
                    "Gemini Query Planner",
                    max_retries=2,
                    base_wait=12,
                    raise_ui_error=False
                )
                parsed = _parse_three_line_plan(raw_out, short_anchor)
                if parsed:
                    return {
                        **parsed,
                        "planner_source": f"gemini:{active_model}",
                        "extracted_opponent_keywords": opp_terms,
                        "planner_errors": planner_errors
                    }
                else:
                    planner_errors.append(f"Gemini planner format mismatch: {raw_out[:120]}")
            except Exception as e:
                planner_errors.append(f"Gemini planner error ({active_model}): {str(e)[:140]}")

    # Stance-aware deterministic fallback if the provider's planner call failed
    if agent_role == "Supporter":
        round_suffixes = {
            1: "efficiency performance gains official statistics dataset",
            2: "cost reduction net benefits lifecycle analysis data",
            3: "long-term stability macro outcomes institutional study"
        }
    else:
        round_suffixes = {
            1: "efficiency loss hidden costs trade-offs empirical study",
            2: "methodology flaws supply bottlenecks adverse impact data",
            3: "infrastructure burden long-term risks viability critique"
        }

    r_suf = round_suffixes.get(current_round, "statistics survey data")
    fallback_tavily = _sanitize_tavily_query(f"{short_anchor} {opp_terms} {r_suf}", short_anchor)
    fallback_exa = f"Peer-reviewed study and statistical findings on {short_anchor} {opp_terms} ({r_suf})"[:350]

    return {
        "inferred_domain": "Universal Fallback",
        "planner_source": f"deterministic_fallback ({agent_role})",
        "tavily_query": fallback_tavily,
        "exa_query": fallback_exa,
        "extracted_opponent_keywords": opp_terms,
        "planner_errors": planner_errors
    }

def fetch_combined_research(
    topic: str,
    agent_role: str,
    current_round: int,
    active_model: str,
    last_opponent_text: str = "",
    seen_urls: list = None,
    seen_titles: list = None,
    past_agent_queries: list = None
) -> tuple[str, dict, list, list]:
    """
    Executes isolated Agentic Query Planning (with cross-round query memory) followed by
    deduplicated Tavily + Exa search.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    exa_key = os.environ.get("EXA_API_KEY", "").strip()

    if not tavily_key and not exa_key:
        raise gr.Error("Missing both TAVILY_API_KEY and EXA_API_KEY in Space Secrets.")

    seen_url_set = set(seen_urls or [])
    seen_title_set = set(seen_titles or [])

    # 1. Agentic Domain Query Formulation (Strictly isolated to the active agent's provider)
    plan = generate_agentic_search_queries(
        topic=topic,
        agent_role=agent_role,
        current_round=current_round,
        active_model=active_model,
        last_opponent_text=last_opponent_text,
        past_agent_queries=past_agent_queries
    )
    t_query = plan["tavily_query"]
    e_query = plan["exa_query"]
    short_anchor = _compress_topic_anchor(topic, max_words=8)

    # Brief 2-second pacing buffer between query planner call and downstream speech call
    time.sleep(2.0)

    snippets = []
    tavily_trace_records = []
    exa_trace_records = []

    # 2. TAVILY SEARCH (Pool of 7 candidates, clean [Title | URL] headers)
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
                ans_text = _clean_snippet_boilerplate(t_resp["answer"])[:380]
                if len(ans_text) >= 80 and "do not provide direct information" not in ans_text.lower():
                    snippets.append(f"- **[Multi-Source Synthesis]:** {ans_text}")
                    tavily_trace_records.append({
                        "type": "tavily_answer",
                        "title": "Multi-Source Synthesis",
                        "url": "N/A",
                        "snippet": ans_text
                    })

            added_tavily = 0
            fallback_tavily_candidates = []
            for res in t_resp.get("results", []):
                url = res.get("url", "")
                raw_title = res.get("title", "Verified Report").strip()
                norm_u = _normalize_url(url)
                norm_t = _normalize_title(raw_title)
                display_title = _clean_display_title(raw_title, url)

                if any(dom in norm_u for dom in EXCLUDED_SEARCH_DOMAINS):
                    continue

                raw_content = res.get("content", "")
                content = _clean_snippet_boilerplate(raw_content)[:480]
                if len(content) < 110:
                    continue

                if norm_u in seen_url_set or (norm_t and norm_t in seen_title_set):
                    fallback_tavily_candidates.append((display_title, url, content, norm_u, norm_t))
                    continue

                seen_url_set.add(norm_u)
                if norm_t:
                    seen_title_set.add(norm_t)
                snippets.append(f"- **[{display_title} | {url}]:** {content}...")
                tavily_trace_records.append({
                    "type": "tavily_result",
                    "title": display_title,
                    "url": url,
                    "snippet": content
                })
                added_tavily += 1
                if added_tavily >= 2:
                    break

            if added_tavily == 0 and fallback_tavily_candidates:
                display_title, url, content, norm_u, norm_t = fallback_tavily_candidates[0]
                snippets.append(f"- **[{display_title} | {url}]:** {content}...")
                tavily_trace_records.append({
                    "type": "tavily_fallback_result",
                    "title": display_title,
                    "url": url,
                    "snippet": content
                })
        except Exception as e:
            err_str = str(e)
            gr.Warning(f"⚠️ Tavily search warning: {err_str}. Continuing with Exa/internal data.")
            tavily_trace_records.append({"error": err_str})

    # 3. EXA NEURAL SEARCH (Pool of 7 candidates, 4-sentence highlights to prevent mid-sentence truncation)
    if exa_key:
        try:
            exa_client = Exa(api_key=exa_key)
            e_resp = exa_client.search_and_contents(
                e_query,
                type="auto",
                num_results=7,
                exclude_domains=EXCLUDED_SEARCH_DOMAINS,
                highlights={
                    "num_sentences": 4,
                    "highlights_per_url": 1,
                    "query": f"{t_query} statistical findings percentages empirical results conclusion"
                },
                text={"max_characters": 650}
            )
            added_exa = 0
            fallback_exa_candidates = []
            for res in getattr(e_resp, "results", []):
                url = getattr(res, "url", "")
                raw_title = (getattr(res, "title", None) or "Academic Study").strip()
                norm_u = _normalize_url(url)
                norm_t = _normalize_title(raw_title)
                display_title = _clean_display_title(raw_title, url)

                if any(dom in norm_u for dom in EXCLUDED_SEARCH_DOMAINS):
                    continue

                highlights_list = getattr(res, "highlights", None) or []
                raw_text = " ".join(highlights_list) if highlights_list else (getattr(res, "text", "") or "")
                text_body = _clean_snippet_boilerplate(raw_text)[:540]
                if len(text_body) < 100:
                    continue

                if norm_u in seen_url_set or (norm_t and norm_t in seen_title_set):
                    fallback_exa_candidates.append((display_title, url, text_body, norm_u, norm_t))
                    continue

                seen_url_set.add(norm_u)
                if norm_t:
                    seen_title_set.add(norm_t)
                snippets.append(f"- **[{display_title} | {url}]:** {text_body}...")
                exa_trace_records.append({
                    "type": "exa_result",
                    "title": display_title,
                    "url": url,
                    "snippet": text_body
                })
                added_exa += 1
                if added_exa >= 2:
                    break

            if added_exa == 0 and fallback_exa_candidates:
                display_title, url, text_body, norm_u, norm_t = fallback_exa_candidates[0]
                snippets.append(f"- **[{display_title} | {url}]:** {text_body}...")
                exa_trace_records.append({
                    "type": "exa_fallback_result",
                    "title": display_title,
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
        "past_queries_avoided": past_agent_queries or [],
        "tavily_dispatched_query": t_query,
        "exa_dispatched_prompt": e_query,
        "tavily_results": tavily_trace_records,
        "exa_results": exa_trace_records,
        "combined_research_char_length": len(combined_text),
        "total_unique_urls_seen_so_far": len(seen_url_set),
        "total_unique_titles_seen_so_far": len(seen_title_set)
    }

    return combined_text, trace_meta, list(seen_url_set), list(seen_title_set)