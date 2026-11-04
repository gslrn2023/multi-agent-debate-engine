# ==============================================================================
# MODULE: config.py
# Version: v2.9.6 | Date: 2026-11-03
# Purpose: Global configuration, constants, CSS, ZeroGPU handler, daily quota
#          limiter, and password gate verification.
# ==============================================================================

import os
import json
import threading
from datetime import datetime, timedelta, timezone
import gradio as gr

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
APP_VERSION = "v2.9.6"
TRACE = True                # // Set True to enable JSON trace logging & UI download button; Set False for production //
DEFAULT_MAX_LOOPS = 3       # // Default & Max number of debate loops/rounds //
DEFAULT_WAIT_TIME_SEC = 30  # // Wait time in seconds between EACH AI agent execution //
MAX_DAILY_DEBATES = 5       # // Global max debates per day across all users //
DAILY_LIMIT_FILE = "daily_limit.json"
DAILY_LIMIT_LOCK = threading.Lock()
IST_TZ = timezone(timedelta(hours=5, minutes=30))

# Prompt files
SUPPORTER_PROMPT_FILE = "supporter_prompt.txt"
CONTENDER_PROMPT_FILE = "contender_prompt.txt"

# Universal list of low-signal social/video/forum/listicle domains excluded from web search
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
    "quora.com",
    "medium.com",
    "gitnux.org",
    "zipdo.co",
    "scribd.com"
]

# Epistemic debate meta-words stripped from Tavily lexical queries to prevent off-topic noise
BANNED_TAVILY_META_WORDS = {
    "empirical", "criticisms", "structural", "limitations", "bottlenecks",
    "discrepancy", "trade-offs", "tradeoffs", "unintended", "consequences",
    "frameworks", "framework", "benchmarks", "validating", "proven",
    "counter-evidence", "counterarguments", "overarching", "comparative",
    "implications", "vulnerabilities", "systemic", "real-world", "opponent"
}

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
    """Checks user password against the CHK_PSSWD secret in Space Settings or local .env."""
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