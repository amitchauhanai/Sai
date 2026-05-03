"""
Sai OS Agent — Cloud Backend v2.0
Production-grade rewrite: Pydantic structured outputs, swarm sub-agents,
SQLite persistent memory, summarization chain, coordinate guardrails.
"""
import logging
import json
import base64
import asyncio
import os
import io
import wave
import time
import math
import sqlite3
import threading
import typing
import urllib.error
import urllib.request
import re
import struct
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import OpenAI
import websockets as ws_client
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, model_validator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sai-server")

app = FastAPI(title="Sai OS Agent Cloud Backend", version="2.0.0")

# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_STT_WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
ELEVENLABS_TTS_VOICE_ID = os.getenv("ELEVENLABS_TTS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVENLABS_TTS_MODEL_ID = os.getenv("ELEVENLABS_TTS_MODEL_ID", "eleven_flash_v2_5")
ELEVENLABS_TTS_OUTPUT_FORMAT = "pcm_16000"
ELEVENLABS_TTS_SAMPLE_RATE = 16000
SAI_VOICE_REPLIES = os.getenv("SAI_VOICE_REPLIES", "true").lower() == "true"
DEEPGRAM_API_KEY = (
    os.getenv("DEEPGRAM_API_KEY")
    or os.getenv("DEEPGRAM_KEY")
    or os.getenv("deep")
)
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_STT_MODEL = os.getenv("GEMINI_STT_MODEL", "gemini-2.5-flash")
STT_PROVIDER = os.getenv(
    "SAI_STT_PROVIDER",
    "deepgram" if DEEPGRAM_API_KEY else ("gemini" if GEMINI_API_KEY else "elevenlabs"),
).strip().lower()
STT_SAMPLE_RATE = 16000
STT_SILENCE_SECS = float(os.getenv("SAI_STT_SILENCE_SECS", "1.1"))
STT_MAX_UTTERANCE_SECS = float(os.getenv("SAI_STT_MAX_UTTERANCE_SECS", "12"))
STT_MIN_SPEECH_SECS = float(os.getenv("SAI_STT_MIN_SPEECH_SECS", "0.45"))
STT_RMS_THRESHOLD = int(os.getenv("SAI_STT_RMS_THRESHOLD", "350"))

nova_client = OpenAI(
    api_key=os.getenv("AMAZON_NOVA_API_KEY"),
    base_url=os.getenv("NOVA_BASE_URL", "https://api.nova.amazon.com/v1"),
)
NOVA_LITE_MODEL_ID = "nova-2-lite-v1"

nova_pro_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    default_headers={
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8080"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "Sai OS Agent"),
    },
)
NOVA_PRO_MODEL_ID = "amazon/nova-pro-v1"

MAX_AGENT_STEPS = 25
LEETCODE_MAX_AGENT_STEPS = 60
ACTION_SETTLE_TIME = 2.0       # seconds to let the UI settle after an action
SCREENSHOT_TIMEOUT = 5.0       # seconds to wait for a screenshot from the client
HISTORY_COMPRESS_THRESHOLD = 9 # compress history when it exceeds this many messages
HISTORY_RECENT_KEEP_PAIRS = 4  # keep the last N user+assistant pairs after compression
ENABLE_CRITIC = os.getenv("SAI_CRITIC_ENABLED", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Pydantic models — every LLM structured output is validated here
# ---------------------------------------------------------------------------

class RoutingDecision(BaseModel):
    complexity: typing.Literal["SIMPLE", "ADVANCED"]
    reason: str


class SimpleAction(BaseModel):
    command: typing.Literal[
        "type_text", "open_app", "open_url", "press_hotkey", "respond", "escalate"
    ]
    text: typing.Optional[str] = None
    app: typing.Optional[str] = None
    url: typing.Optional[str] = None
    browser: typing.Optional[str] = None
    keys: typing.Optional[list[str]] = None


class AgentAction(BaseModel):
    explanation: str
    command: typing.Literal[
        "click", "type_text", "open_app", "keyboard_type", "open_url",
        "press_hotkey", "scroll", "wait",
    ]
    x: typing.Optional[int] = Field(default=None, ge=0, le=1000)
    y: typing.Optional[int] = Field(default=None, ge=0, le=1000)
    text: typing.Optional[str] = None
    app: typing.Optional[str] = None
    url: typing.Optional[str] = None
    browser: typing.Optional[str] = None
    keys: typing.Optional[list[str]] = None
    amount: typing.Optional[int] = None
    done: bool = False

    @model_validator(mode="after")
    def click_requires_coords(self) -> "AgentAction":
        if self.command == "click" and (self.x is None or self.y is None):
            raise ValueError("click command requires both x and y coordinates in [0, 1000]")
        return self


class CriticVerdict(BaseModel):
    approved: bool
    reason: str
    corrected_x: typing.Optional[int] = Field(default=None, ge=0, le=1000)
    corrected_y: typing.Optional[int] = Field(default=None, ge=0, le=1000)


class MemoryContext(BaseModel):
    relevant_facts: list[str] = Field(default_factory=list)
    session_summary: str = ""


class ConversationSummary(BaseModel):
    summary: str


class IntentResult(BaseModel):
    corrected_command: str

# ---------------------------------------------------------------------------
# Persistent memory — SQLite at ~/.sai/memory.db
# ---------------------------------------------------------------------------

_MEMORY_DB_PATH = Path.home() / ".sai" / "memory.db"
_db_lock = threading.Lock()
_memory_db: typing.Optional[sqlite3.Connection] = None


def _get_db() -> sqlite3.Connection:
    global _memory_db
    if _memory_db is None:
        _MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_MEMORY_DB_PATH), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT    NOT NULL,
                task       TEXT,
                outcome    TEXT,
                summary    TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT    NOT NULL,
                category   TEXT    NOT NULL,
                content    TEXT    NOT NULL UNIQUE
            )
        """)
        conn.commit()
        _memory_db = conn
    return _memory_db


def store_session(task: str, outcome: str, summary: str) -> None:
    with _db_lock:
        _get_db().execute(
            "INSERT INTO sessions (started_at, task, outcome, summary) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), task, outcome, summary),
        )
        _get_db().commit()


def fetch_recent_sessions(limit: int = 5) -> list[dict]:
    with _db_lock:
        rows = _get_db().execute(
            "SELECT task, outcome, summary FROM sessions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"task": r[0], "outcome": r[1], "summary": r[2]} for r in rows]


def store_fact(category: str, content: str) -> None:
    with _db_lock:
        _get_db().execute(
            "INSERT OR IGNORE INTO facts (created_at, category, content) VALUES (?, ?, ?)",
            (datetime.utcnow().isoformat(), category, content),
        )
        _get_db().commit()


def fetch_facts(limit: int = 20) -> list[dict]:
    with _db_lock:
        rows = _get_db().execute(
            "SELECT category, content FROM facts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"category": r[0], "content": r[1]} for r in rows]

# ---------------------------------------------------------------------------
# Structured LLM call helper
# ---------------------------------------------------------------------------

def _extract_first_json_object(raw: str) -> str:
    """Return the first balanced JSON object from model output."""
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(
            line for line in clean.splitlines()
            if not line.startswith("```")
        ).strip()

    start = clean.find("{")
    if start == -1:
        return clean

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(clean)):
        char = clean[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return clean[start : index + 1]

    return clean[start:]


def _call_structured_sync(
    client: OpenAI,
    model: str,
    messages: list[dict],
    response_model: type[BaseModel],
    temperature: float = 0,
) -> BaseModel:
    """
    Synchronous wrapper: calls the LLM and validates the response into a Pydantic model.

    Strategy:
    1. Request JSON mode (response_format=json_object) when the endpoint supports it.
    2. Fall back to a plain call if JSON mode is unsupported.
    3. In both cases, extract the JSON substring from the raw content, then
       validate it strictly with Pydantic. No regex hacks — if parsing fails
       we raise so the caller can decide how to recover.
    """
    # Attempt JSON mode first
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Endpoint doesn't support response_format — plain call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    raw = response.choices[0].message.content.strip()

    raw = _extract_first_json_object(raw)

    try:
        return response_model.model_validate_json(raw)
    except Exception as exc:
        raise ValueError(
            f"Pydantic validation failed for {response_model.__name__}: {exc} | raw={raw[:300]}"
        ) from exc


async def _call_structured(
    client: OpenAI,
    model: str,
    messages: list[dict],
    response_model: type[BaseModel],
    temperature: float = 0,
) -> BaseModel:
    """Async shim: runs the blocking OpenAI call in the default thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _call_structured_sync(client, model, messages, response_model, temperature),
    )


def strip_wake_phrase(text: str) -> str:
    """Manual mode already handles waking; remove spoken wake words from commands."""
    cleaned = re.sub(
        r"^\s*(?:hey|hi|hello)?\s*,?\s*sai(?:ther)?\s*[,.\-:]*\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or text.strip()


STOP_COMMANDS = {
    "stop",
    "quit",
    "exit",
    "shutdown",
    "shut down",
    "stop sai",
}


def is_stop_command(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized in STOP_COMMANDS


KNOWN_SITE_ALIASES = {
    "amazon": "https://www.amazon.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "figma": "https://www.figma.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "gioogle": "https://www.google.com",
    "gogle": "https://www.google.com",
    "goodgle": "https://www.google.com",
    "google": "https://www.google.com",
    "google docs": "https://docs.google.com",
    "google drive": "https://drive.google.com",
    "google maps": "https://maps.google.com",
    "google sheets": "https://sheets.google.com",
    "google slides": "https://slides.google.com",
    "lead code": "https://leetcode.com",
    "leedcode": "https://leetcode.com",
    "leedcofe": "https://leetcode.com",
    "leetcode": "https://leetcode.com",
    "leetcofe": "https://leetcode.com",
    "leet code": "https://leetcode.com",
    "linkedin": "https://www.linkedin.com",
    "netflix": "https://www.netflix.com",
    "openai": "https://openai.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "youtube": "https://www.youtube.com",
}

KNOWN_APP_ALIASES = {
    "arc": "Arc",
    "brave": "Brave Browser",
    "brave browser": "Brave Browser",
    "calculator": "Calculator",
    "calendar": "Calendar",
    "chrome": "Google Chrome",
    "discord": "Discord",
    "edge": "Microsoft Edge",
    "finder": "Finder",
    "firefox": "Firefox",
    "mail": "Mail",
    "messages": "Messages",
    "notes": "Notes",
    "safari": "Safari",
    "slack": "Slack",
    "spotify": "Spotify",
    "terminal": "Terminal",
    "textedit": "TextEdit",
    "text edit": "TextEdit",
    "visual studio code": "Visual Studio Code",
    "vs code": "Visual Studio Code",
}

BROWSER_APP_NAMES = {
    "Arc",
    "Brave Browser",
    "Chromium",
    "Firefox",
    "Google Chrome",
    "Microsoft Edge",
    "Opera",
    "Safari",
    "Vivaldi",
}

URLISH_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?::\d+)?(?:[/?#][^\s]*)?$",
    re.IGNORECASE,
)
SINGLE_HOST_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$", re.IGNORECASE)
OPEN_URL_PREFIX_RE = re.compile(
    r"^(?:open|go to|goto|visit|browse to|navigate to|take me to)\s+(.+?)\s*$",
    re.IGNORECASE,
)
OPEN_APP_PREFIX_RE = re.compile(r"^(?:open|launch|start)\s+(.+?)\s*$", re.IGNORECASE)
OPEN_APP_ANYWHERE_RE = re.compile(
    r"\b(?:open|launch|start)\s+(safari|chrome|google chrome|arc|firefox|brave|brave browser|edge|microsoft edge)\b",
    re.IGNORECASE,
)
OPEN_APP_BROWSER_HINT_RE = re.compile(
    r"\b(?:in|on|with|using)\s+(safari|chrome|google chrome|arc|firefox|brave|brave browser|edge|microsoft edge)\b",
    re.IGNORECASE,
)
OPEN_IN_TAB_RE = re.compile(
    r"\b(?:in|into|on|with)\s+(?:a\s+|the\s+)?(?:new\s+)?(?:browser\s+)?tab\b",
    re.IGNORECASE,
)
TAB_TARGET_PREFIX_RE = re.compile(
    r"^(?:a\s+|the\s+)?(?:new\s+)?(?:browser\s+)?tab\s+(?:for|to|with)\s+",
    re.IGNORECASE,
)
MULTI_STEP_MARKERS = (
    " and ",
    " then ",
    " after ",
    " before ",
    " once ",
    " while ",
    " type ",
    " click ",
    " scroll ",
    " log in",
    " sign in",
    " fill ",
    " enter ",
    " turn ",
    " toggle ",
    " submit ",
)

TAB_HOTKEY_ACTIONS: tuple[tuple[tuple[str, ...], dict], ...] = (
    (
        (
            "reopen closed tab",
            "reopen last tab",
            "restore closed tab",
            "restore last tab",
            "open last closed tab",
        ),
        {"command": "press_hotkey", "keys": ["command", "shift", "t"]},
    ),
    (
        (
            "close tab",
            "close current tab",
            "close this tab",
            "close browser tab",
        ),
        {"command": "press_hotkey", "keys": ["command", "w"]},
    ),
    (
        (
            "next tab",
            "switch to next tab",
            "go to next tab",
            "move to next tab",
            "right tab",
        ),
        {"command": "press_hotkey", "keys": ["ctrl", "tab"]},
    ),
    (
        (
            "previous tab",
            "prev tab",
            "switch to previous tab",
            "go to previous tab",
            "move to previous tab",
            "left tab",
        ),
        {"command": "press_hotkey", "keys": ["ctrl", "shift", "tab"]},
    ),
    (
        (
            "new tab",
            "open new tab",
            "open a new tab",
            "open the new tab",
            "open tab",
            "open another tab",
            "create tab",
            "create new tab",
            "make new tab",
            "new browser tab",
            "open browser tab",
        ),
        {"command": "press_hotkey", "keys": ["command", "t"]},
    ),
)


def _clean_simple_target(raw: str) -> str:
    target = (raw or "").strip().strip("\"'")
    target = re.sub(r"^(?:the)\s+", "", target, flags=re.IGNORECASE)
    target = re.sub(
        r"\b(?:website|site|homepage|home page|page)\b",
        "",
        target,
        flags=re.IGNORECASE,
    )
    target = re.sub(r"\s+", " ", target)
    return target.strip(" .,!?:;")


def _spoken_target_to_urlish(target: str) -> str:
    urlish = target.lower()
    replacements = (
        (" dot ", "."),
        (" slash ", "/"),
        (" backslash ", "/"),
        (" colon ", ":"),
        (" question mark ", "?"),
        (" hashtag ", "#"),
    )
    for old, new in replacements:
        urlish = urlish.replace(old, new)
    urlish = re.sub(r"\s*\.\s*", ".", urlish)
    urlish = re.sub(r"\s*/\s*", "/", urlish)
    urlish = re.sub(r"\s*:\s*", ":", urlish)
    return urlish.strip()


def _strip_search_suffix(raw: str) -> str:
    return re.sub(
        r"\b(?:search|search engine|site|website|homepage|home page|page)\b\.?$",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()


def normalize_simple_url_target(raw: str) -> typing.Optional[str]:
    target = _clean_simple_target(raw)
    if not target:
        return None

    lowered = target.lower()
    if lowered in KNOWN_SITE_ALIASES:
        return KNOWN_SITE_ALIASES[lowered]

    if lowered.endswith(" search"):
        shortened = lowered[: -len(" search")].strip()
        if shortened in KNOWN_SITE_ALIASES:
            return KNOWN_SITE_ALIASES[shortened]

    urlish = _spoken_target_to_urlish(target)
    compact = re.sub(r"\s+", "", urlish)
    if compact in KNOWN_SITE_ALIASES:
        return KNOWN_SITE_ALIASES[compact]

    if URLISH_RE.fullmatch(compact):
        if compact.startswith(("http://", "https://")):
            return compact
        return f"https://{compact}"

    return None


def strip_tab_target_phrasing(raw: str) -> str:
    target = TAB_TARGET_PREFIX_RE.sub("", raw or "")
    target = OPEN_IN_TAB_RE.sub("", target)
    return _clean_simple_target(target)


def normalize_app_target(raw: str) -> typing.Optional[str]:
    target = _clean_simple_target(raw)
    if not target:
        return None

    lowered = target.lower()
    if lowered in KNOWN_APP_ALIASES:
        return KNOWN_APP_ALIASES[lowered]
    if lowered in KNOWN_SITE_ALIASES:
        return None
    if " " not in lowered and SINGLE_HOST_RE.fullmatch(lowered):
        return target
    if lowered.endswith(" app"):
        return target[:-4].strip()
    return None


def deterministic_tab_action(user_text: str) -> typing.Optional[dict]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", user_text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    article_free = re.sub(r"\b(?:a|the)\b", "", normalized)
    article_free = re.sub(r"\s+", " ", article_free).strip()
    for phrases, action in TAB_HOTKEY_ACTIONS:
        if normalized in phrases or article_free in phrases:
            return action.copy()
    return None


def deterministic_simple_action(user_text: str) -> typing.Optional[dict]:
    """
    Fast-path obvious app launches and website opens without waiting on an LLM.
    This makes commands like "open Google" reliable even if the router is flaky.
    """
    normalized = re.sub(r"\s+", " ", user_text.lower()).strip()
    padded = f" {normalized} "

    if action := deterministic_tab_action(user_text):
        return action

    if any(marker in padded for marker in MULTI_STEP_MARKERS):
        return None

    if url_match := OPEN_URL_PREFIX_RE.match(user_text):
        target = strip_tab_target_phrasing(url_match.group(1))
        if url := normalize_simple_url_target(target):
            return {"command": "open_url", "url": url}

    if app_match := OPEN_APP_PREFIX_RE.match(user_text):
        target = strip_tab_target_phrasing(app_match.group(1))
        if url := normalize_simple_url_target(target):
            return {"command": "open_url", "url": url}
        if app_name := normalize_app_target(target):
            return {"command": "open_app", "app": app_name}

    return None


def _known_site_actions_in_order(
    normalized: str,
    browser: typing.Optional[str] = None,
) -> list[dict]:
    matches: list[tuple[int, int, str, str]] = []
    for phrase, url in KNOWN_SITE_ALIASES.items():
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", normalized):
            matches.append((match.start(), match.end(), phrase, url))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

    occupied_spans: list[tuple[int, int]] = []
    seen_urls: set[str] = set()
    actions: list[dict] = []
    for start, end, _phrase, url in matches:
        if any(start < span_end and end > span_start for span_start, span_end in occupied_spans):
            continue
        occupied_spans.append((start, end))
        if url in seen_urls:
            continue
        seen_urls.add(url)
        actions.append({
            "command": "open_url",
            "url": url,
            **({"browser": browser} if browser in BROWSER_APP_NAMES else {}),
        })
    return actions


def deterministic_start_actions(user_text: str) -> list[dict]:
    """
    Pull obvious "open browser/site first" work out of multi-step commands.

    Voice requests such as "open Safari and login to my LeetCode" should first
    leave VS Code and put the requested browser/site in front. The vision loop
    can continue afterward only if there is still real UI work to do.
    """
    normalized = re.sub(r"\s+", " ", user_text.lower()).strip()
    actions: list[dict] = []
    browser: typing.Optional[str] = None

    if app_match := OPEN_APP_ANYWHERE_RE.search(normalized):
        browser = normalize_app_target(app_match.group(1))
        if browser:
            actions.append({"command": "open_app", "app": browser})
    elif browser_match := OPEN_APP_BROWSER_HINT_RE.search(normalized):
        browser = normalize_app_target(browser_match.group(1))
        if browser:
            actions.append({"command": "open_app", "app": browser})

    # Login/sign-in requests should land on the login page when we know it.
    leetcode_login = is_leetcode_task(user_text) and re.search(
        r"\b(?:log\s*in|login|sign\s*in|signin)\b", normalized
    )
    if leetcode_login:
        actions.append({
            "command": "open_url",
            "url": "https://leetcode.com/accounts/login/",
            **({"browser": browser} if browser in BROWSER_APP_NAMES else {}),
        })
        return _dedupe_actions(actions)

    if is_leetcode_task(user_text):
        if problem_url := leetcode_problem_url(user_text):
            actions.append({
                "command": "open_url",
                "url": problem_url,
                **({"browser": browser} if browser in BROWSER_APP_NAMES else {}),
            })
            return _dedupe_actions(actions)

    if re.search(r"\b(?:open|go to|goto|visit|browse to|navigate to|take me to)\b", normalized):
        site_actions = _known_site_actions_in_order(normalized, browser)
        if site_actions:
            actions.extend(site_actions)
        elif is_leetcode_task(user_text):
            actions.append({
                "command": "open_url",
                "url": "https://leetcode.com",
                **({"browser": browser} if browser in BROWSER_APP_NAMES else {}),
            })

    return _dedupe_actions(actions)


def _dedupe_actions(actions: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for action in actions:
        key = tuple(
            sorted(
                (item_key, tuple(item_value) if isinstance(item_value, list) else item_value)
                for item_key, item_value in action.items()
            )
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def is_leetcode_task(user_text: str) -> bool:
    lowered = user_text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "leetcode",
            "lead code",
            "leedcode",
            "leedcofe",
            "leetcofe",
            "leet code",
            "daily problem",
            "daily challenge",
            "solve today's daily",
            "solve todays daily",
        )
    )


def leetcode_problem_url(user_text: str) -> typing.Optional[str]:
    """Extract a direct LeetCode problem URL from a spoken problem reference."""
    normalized = user_text.lower()
    # Specific quoted problem titles: problem "1. Two Sum" or problem 'two sum'
    quoted = re.search(r"problem\s+[\"']([^\"']+)[\"']", user_text, re.IGNORECASE)
    if quoted:
        title = quoted.group(1)
    else:
        numbered = re.search(
            r"problem\s+(?:#?\s*)?(\d+)\.?\s*([a-z0-9][a-z0-9\s\-']*)",
            normalized,
        )
        if numbered:
            title = numbered.group(2)
        else:
            return None

    title = title.strip().lower()
    title = re.sub(r"^\s*\d+\s*\.?\s*", "", title)
    title = re.sub(r"['’]", "", title)
    slug = re.sub(r"[^a-z0-9]+", "-", title).strip("-")
    if not slug:
        return None
    return f"https://leetcode.com/problems/{slug}/"


def should_continue_after_start_actions(user_text: str) -> bool:
    normalized = user_text.lower()
    return bool(re.search(
        r"\b(?:solve|submit|answer|navigate to problem|open problem|leetcode problem|practice|attempt|debug|problem)\b",
        normalized,
    ))


def task_specific_guidance(user_text: str) -> str:
    if not is_leetcode_task(user_text):
        return ""

    return """
LEETCODE MODE:
- The task is only complete when the screenshot clearly shows an accepted/successful submission for the requested problem.
- If the browser or site is not open yet, you may use open_url to reach Google, LeetCode, or the daily challenge page directly.
- Read the problem statement, constraints, examples, and the currently-selected language before writing code.
- Prefer replacing the full editor contents with a complete working solution instead of patching tiny fragments.
- After every submission, inspect the visible result carefully.
- If you see Wrong Answer, Runtime Error, Compile Error, Memory Limit Exceeded, Time Limit Exceeded, or a failing testcase, diagnose the issue from the visible feedback, revise the code, and submit again.
- Use wait when the judge is still running or the page is loading.
- Never set done=true until the screenshot clearly confirms Accepted or an equivalent success state.
""".strip()


def local_simple_response(user_text: str) -> typing.Optional[str]:
    """Answer tiny local utility queries without an LLM or UI action."""
    normalized = re.sub(r"[^a-z0-9\s]", " ", user_text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return "I'm listening. What would you like me to do?"

    if normalized in {"hey sai", "hi sai", "hello sai", "sai"}:
        return "I'm listening. Tell me the task after you press Enter."

    time_queries = {
        "time",
        "current time",
        "what time is it",
        "tell me the time",
        "what is the time",
    }
    if normalized in time_queries:
        return "The time is " + datetime.now().strftime("%I:%M %p").lstrip("0") + "."

    date_queries = {
        "date",
        "day",
        "today s date",
        "today date",
        "what day is it",
        "what is today",
        "what is the date",
        "what date is it",
        "tell me the date",
    }
    if normalized in date_queries:
        return "Today is " + datetime.now().strftime("%A, %B %d, %Y") + "."

    if "weather" in normalized:
        return (
            "I can help with weather, but I need a city or a weather API. "
            "Try saying: open weather dot com, or ask for the weather in a city."
        )

    return None

# ---------------------------------------------------------------------------
# Screenshot annotation
# ---------------------------------------------------------------------------

def annotate_screenshot(
    image_b64: str,
    last_action: typing.Optional[dict] = None,
) -> str:
    """
    Draw ruler tick marks on the screenshot edges and optionally a crosshair
    at the last click position.  All coordinates are treated as normalized
    [0, 1000] values and mapped to actual image pixels — the image may be
    any resolution.
    """
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        except Exception:
            font = ImageFont.load_default()

        TICK_LEN = 12
        LABEL_PAD = 2
        TICK_STEPS = 5          # marks at 0, 200, 400, 600, 800, 1000
        TICK_COLOR = (255, 60, 60)
        LABEL_COLOR = (255, 255, 255)
        step_norm = 1000 // TICK_STEPS

        for i in range(1, TICK_STEPS + 1):
            norm = i * step_norm
            # Top-edge (X-axis)
            px = int(round((norm / 1000) * w))
            draw.line([(px, 0), (px, TICK_LEN)], fill=TICK_COLOR, width=2)
            draw.text((px + LABEL_PAD, LABEL_PAD), str(norm), fill=LABEL_COLOR, font=font)
            # Left-edge (Y-axis)
            py = int(round((norm / 1000) * h))
            draw.line([(0, py), (TICK_LEN, py)], fill=TICK_COLOR, width=2)
            draw.text((LABEL_PAD, py + LABEL_PAD), str(norm), fill=LABEL_COLOR, font=font)

        # Last-click crosshair (normalized → pixel)
        if last_action and last_action.get("command") == "click":
            norm_x = float(last_action.get("x", 0))
            norm_y = float(last_action.get("y", 0))
            px = int(round((norm_x / 1000) * w))
            py = int(round((norm_y / 1000) * h))
            r = 20
            draw.ellipse([px - r, py - r, px + r, py + r], outline="lime", width=3)
            draw.line([px - r * 2, py, px + r * 2, py], fill="lime", width=2)
            draw.line([px, py - r * 2, px, py + r * 2], fill="lime", width=2)
            draw.text(
                (px + r + 5, py - 10),
                f"CLICKED ({int(norm_x)},{int(norm_y)})",
                fill="lime",
                font=font,
            )

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        logger.error(f"annotate_screenshot failed: {exc}")
        return image_b64

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Sai OS Agent Cloud Backend is running",
        "version": "2.0.0",
        "critic_enabled": ENABLE_CRITIC,
        "memory_db": str(_MEMORY_DB_PATH),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "websocket": "/ws/agent",
        "stt_provider": STT_PROVIDER,
        "voice_replies": SAI_VOICE_REPLIES,
    }


def synthesize_speech_pcm(text: str) -> bytes:
    """Generate 16 kHz signed 16-bit PCM audio for local playback."""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_TTS_VOICE_ID}"
        f"?output_format={ELEVENLABS_TTS_OUTPUT_FORMAT}"
    )
    payload = json.dumps({
        "text": text,
        "model_id": ELEVENLABS_TTS_MODEL_ID,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.75,
            "speed": 1.04,
        },
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ElevenLabs TTS failed: {exc.code} {detail}") from exc


def _pcm16_rms(pcm: bytes) -> float:
    """Compute RMS volume for signed 16-bit little-endian mono PCM."""
    if len(pcm) < 2:
        return 0.0
    usable = len(pcm) - (len(pcm) % 2)
    samples = struct.unpack("<" + "h" * (usable // 2), pcm[:usable])
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = STT_SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def transcribe_speech_with_gemini(pcm: bytes) -> str:
    """Transcribe a completed 16 kHz PCM utterance with Gemini audio understanding."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    wav_bytes = _pcm16_to_wav_bytes(pcm)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_STT_MODEL}:generateContent"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Transcribe the spoken command in this audio. "
                            "Return only the words the user said. "
                            "If there is no clear speech, return an empty string."
                        )
                    },
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": base64.b64encode(wav_bytes).decode("utf-8"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 80,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini STT failed: {exc.code} {detail}") from exc

    parts = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    text = " ".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and part.get("text")
    )
    return text.strip().strip('"').strip("'")


@app.websocket("/ws/agent")
async def websocket_endpoint(websocket: WebSocket):  # noqa: C901
    await websocket.accept()
    logger.info("New client connection established")

    state: dict[str, typing.Any] = {
        "latest_screenshot_b64": None,
        "latest_app_context": {},
        "screen_width": None,           # physical pixel dimensions reported by client
        "screen_height": None,
        "transcription_buffer": [],
        "debounce_task": None,
        "active_agent_task": None,
        "command_triggered": False,
        "manual_text_command": None,
        "screenshot_event": asyncio.Event(),
    }

    if STT_PROVIDER == "deepgram" and not DEEPGRAM_API_KEY:
        logger.error("DEEPGRAM_API_KEY not set")
        await websocket.close(code=1011)
        return
    if STT_PROVIDER == "gemini" and not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set")
        await websocket.close(code=1011)
        return
    if STT_PROVIDER == "elevenlabs" and not ELEVENLABS_API_KEY:
        logger.error("ELEVENLABS_API_KEY not set")
        await websocket.close(code=1011)
        return

    try:
        # ------------------------------------------------------------------
        # Handshake
        # ------------------------------------------------------------------
        initial_data = await websocket.receive_text()
        event_payload = json.loads(initial_data)
        if event_payload.get("event") != "wake_word_detected":
            await websocket.close(code=1003)
            return
        state["manual_text_command"] = (event_payload.get("text_command") or "").strip()

        logger.info("Handshake successful")
        await websocket.send_json({"status": "handshake_complete"})
        await websocket.send_text(json.dumps({"command": "set_activity", "state": "active"}))

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _app_ctx_summary() -> str:
            ctx = state.get("latest_app_context", {})
            app_name = ctx.get("app_name", "")
            if not app_name:
                return "No active app information available."
            parts = [f"Active app: {app_name}"]
            if url := ctx.get("tab_url", ""):
                parts.append(f"Browser tab URL: {url}")
            if title := ctx.get("tab_title", ""):
                parts.append(f"Tab title: {title}")
            return " | ".join(parts)

        async def send_voice_reply(text: str) -> None:
            if not SAI_VOICE_REPLIES:
                return
            try:
                loop = asyncio.get_running_loop()
                pcm = await loop.run_in_executor(None, synthesize_speech_pcm, text)
                await websocket.send_text(json.dumps({
                    "command": "speak_audio",
                    "audio_base64": base64.b64encode(pcm).decode("utf-8"),
                    "audio_format": ELEVENLABS_TTS_OUTPUT_FORMAT,
                    "sample_rate": ELEVENLABS_TTS_SAMPLE_RATE,
                }))
            except Exception as exc:
                logger.warning(f"Voice reply failed: {exc}")

        # ------------------------------------------------------------------
        # Sub-Agent 1 — Memory Fetcher
        # Runs once per task before the Senior Brain loop starts.
        # Retrieves and ranks relevant past sessions and stored facts.
        # ------------------------------------------------------------------

        async def memory_sub_agent(task: str) -> MemoryContext:
            recent = fetch_recent_sessions(limit=5)
            facts = fetch_facts(limit=20)
            if not recent and not facts:
                return MemoryContext()

            prompt = (
                f'You are the Memory Sub-Agent for Sai, a macOS desktop assistant.\n'
                f'Given the CURRENT TASK and past memory, identify what is RELEVANT.\n\n'
                f'CURRENT TASK: "{task}"\n\n'
                f'PAST SESSIONS:\n{json.dumps(recent, indent=2)}\n\n'
                f'STORED FACTS:\n{json.dumps(facts, indent=2)}\n\n'
                'Output JSON:\n'
                '- relevant_facts: list of strings (max 5) — concise, directly relevant facts\n'
                '- session_summary: one sentence of the most relevant past context, or ""\n'
                'If nothing is relevant, return empty values.'
            )
            try:
                result = await _call_structured(
                    nova_client,
                    NOVA_LITE_MODEL_ID,
                    [
                        {"role": "system", "content": "Output JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    MemoryContext,
                )
                logger.info(
                    f"Memory sub-agent: {len(result.relevant_facts)} facts, "
                    f"summary={'yes' if result.session_summary else 'none'}"
                )
                return result
            except Exception as exc:
                logger.error(f"Memory sub-agent failed: {exc}")
                return MemoryContext()

        # ------------------------------------------------------------------
        # Sub-Agent 2 — Critic
        # Intercepts every planned click and verifies the coordinate lands on
        # an interactive element in the annotated screenshot.  Uses Nova Lite
        # for minimal latency overhead.
        # ------------------------------------------------------------------

        async def critic_sub_agent(
            action: AgentAction,
            annotated_b64: str,
        ) -> CriticVerdict:
            if not ENABLE_CRITIC or action.command != "click":
                return CriticVerdict(approved=True, reason="Critic skipped.")

            prompt = (
                f"You are a UI Critic verifying a planned click.\n"
                f"Agent plans to click at normalized ({action.x}, {action.y}) on a 0-1000 grid.\n"
                f"Agent reasoning: {action.explanation}\n\n"
                "The screenshot has ruler tick marks on the top and left edges at 200, 400, 600, 800, 1000.\n"
                "Use them to estimate where the click will land.\n\n"
                "Does this coordinate land on an interactive element (button, link, input, tab, etc.)?\n\n"
                "Output JSON:\n"
                "- approved: true if the click looks correct\n"
                "- reason: brief justification (one sentence)\n"
                "- corrected_x / corrected_y: if approved=false and you can see the correct target, "
                "provide corrected coords in [0,1000]; otherwise null"
            )
            try:
                result = await _call_structured(
                    nova_client,
                    NOVA_LITE_MODEL_ID,
                    [
                        {"role": "system", "content": "Output JSON only."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{annotated_b64}"},
                                },
                            ],
                        },
                    ],
                    CriticVerdict,
                )
                logger.info(
                    f"Critic: approved={result.approved} | {result.reason[:80]}"
                )
                return result
            except Exception as exc:
                logger.error(f"Critic sub-agent failed, approving by default: {exc}")
                return CriticVerdict(approved=True, reason="Critic unavailable.")

        # ------------------------------------------------------------------
        # Summarization chain
        # When conversation_history exceeds the threshold, older exchanges are
        # compressed by Nova Lite into a single context block.  This preserves
        # the overarching goal across long multi-step tasks while keeping the
        # context window lean.
        # ------------------------------------------------------------------

        async def _compress_history(
            history: list[dict],
            task: str,
        ) -> list[dict]:
            if len(history) <= HISTORY_COMPRESS_THRESHOLD:
                return history

            system_msgs = [m for m in history if m["role"] == "system"]
            non_system = [m for m in history if m["role"] != "system"]

            keep_count = HISTORY_RECENT_KEEP_PAIRS * 2
            to_summarize = non_system[:-keep_count] if len(non_system) > keep_count else []
            recent = non_system[-keep_count:]

            if not to_summarize:
                return history

            # Build text-only digest (drop image parts to keep prompt small)
            lines: list[str] = []
            for msg in to_summarize:
                role = msg["role"].upper()
                content = msg["content"]
                if isinstance(content, list):
                    text_parts = [
                        p["text"]
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = " ".join(text_parts)
                lines.append(f"{role}: {str(content)[:400]}")

            prompt = (
                f"Summarize the following agent steps as a compact context block.\n"
                f"Overarching task: '{task}'\n"
                f"Preserve: what was tried, what worked or failed, and the current progress.\n\n"
                + "\n".join(lines)
            )
            try:
                result = await _call_structured(
                    nova_client,
                    NOVA_LITE_MODEL_ID,
                    [
                        {"role": "system", "content": "Output JSON with a single 'summary' key."},
                        {"role": "user", "content": prompt},
                    ],
                    ConversationSummary,
                )
                summary_msg = {
                    "role": "user",
                    "content": f"[COMPRESSED HISTORY — {len(to_summarize)} steps]\n{result.summary}",
                }
                logger.info(
                    f"History compressed: {len(to_summarize)} messages → 1 summary block"
                )
                return system_msgs + [summary_msg] + recent
            except Exception as exc:
                logger.error(f"Summarization failed, falling back to naive trim: {exc}")
                return system_msgs + non_system[:3] + recent

        # ------------------------------------------------------------------
        # Coordinate safety guardrail
        # ------------------------------------------------------------------

        def _coords_valid(action: AgentAction) -> bool:
            if action.command != "click":
                return True
            return (
                action.x is not None
                and action.y is not None
                and 0 <= action.x <= 1000
                and 0 <= action.y <= 1000
            )

        # ------------------------------------------------------------------
        # Intent interpretation
        # ------------------------------------------------------------------

        async def interpret_intent(raw: str) -> str:
            prompt = (
                f"You are a voice command interpreter for Sai, a macOS desktop assistant.\n"
                f"SCREEN CONTEXT: {_app_ctx_summary()}\n"
                f'RAW TRANSCRIPTION: "{raw}"\n\n'
                "Reconstruct the user's ACTUAL INTENDED COMMAND from the (possibly garbled) transcription.\n"
                "Common STT errors: 'they\\'re not'→'turn off', 'clothes'→'close', 'right'→'write'.\n"
                "Output ONLY JSON: {\"corrected_command\": \"...\"}"
            )
            try:
                result = await _call_structured(
                    nova_client,
                    NOVA_LITE_MODEL_ID,
                    [
                        {"role": "system", "content": "Output JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    IntentResult,
                )
                corrected = result.corrected_command.strip().strip('"').strip("'")
                if len(corrected) > 2:
                    logger.info(f"Intent: '{raw}' → '{corrected}'")
                    return corrected
            except Exception as exc:
                logger.error(f"Intent interpretation failed, using raw: {exc}")
            return raw

        # ------------------------------------------------------------------
        # Hybrid router
        # ------------------------------------------------------------------

        async def hybrid_reasoning(user_text: str) -> typing.Optional[dict]:
            ctx = _app_ctx_summary()

            if action := deterministic_simple_action(user_text):
                logger.info(f"Bypassing router with deterministic action: {action}")
                return action

            routing_prompt = (
                f"You are the Task Router for Sai, a macOS desktop assistant.\n"
                f"SCREEN CONTEXT: {ctx}\n\n"
                "SIMPLE = single fire-and-forget action that does NOT need to see the screen:\n"
                "  - Launch an app via Spotlight\n"
                "  - Open a brand-new URL the user is NOT already on\n"
                "  - A single global hotkey\n\n"
                "ADVANCED = anything requiring screen interaction, multi-step UI navigation,\n"
                "or any task relating to the app/site currently open.\n\n"
                f'User said: "{user_text}"\n\n'
                'Output JSON: {"complexity": "SIMPLE" | "ADVANCED", "reason": "one sentence"}'
            )

            complexity = "ADVANCED"
            try:
                decision = await _call_structured(
                    nova_client,
                    NOVA_LITE_MODEL_ID,
                    [
                        {"role": "system", "content": "Output JSON only."},
                        {"role": "user", "content": routing_prompt},
                    ],
                    RoutingDecision,
                )
                complexity = decision.complexity
                logger.info(f"Routing: {complexity} | {decision.reason}")
            except Exception as exc:
                logger.error(f"Routing failed, defaulting to ADVANCED: {exc}")

            if complexity == "SIMPLE":
                simple_prompt = (
                    "Convert the user's request to a SINGLE tool-call JSON.\n"
                    "AVAILABLE COMMANDS:\n"
                    '  {"command": "open_app", "app": "AppName"}  — launch a macOS app directly\n'
                    '  {"command": "type_text", "text": "AppName"}  — legacy Spotlight app launch fallback only\n'
                    '  {"command": "open_url", "url": "https://...", "browser": "Safari"}  — open a URL, browser optional\n'
                    '  {"command": "press_hotkey", "keys": ["command", "n"]}  — keyboard shortcut\n'
                    '  {"command": "respond", "text": "..."}  — answer simple conversational questions\n'
                    '  {"command": "escalate"}  — if on-screen interaction is required\n\n'
                    f'User request: "{user_text}"\n'
                    "Output JSON only."
                )
                try:
                    action = await _call_structured(
                        nova_client,
                        NOVA_LITE_MODEL_ID,
                        [
                            {"role": "system", "content": "Output JSON only."},
                            {"role": "user", "content": simple_prompt},
                        ],
                        SimpleAction,
                    )
                    if action.command != "escalate":
                        logger.info(f"SIMPLE action: {action.model_dump(exclude_none=True)}")
                        if action.command == "respond":
                            await send_voice_reply(action.text or "Done.")
                            return None
                        return action.model_dump(exclude_none=True)
                    logger.info("SIMPLE handler escalated to ADVANCED.")
                except Exception as exc:
                    logger.error(f"SIMPLE generation failed, escalating: {exc}")

            # ADVANCED — start the swarm agent loop
            if (
                state["active_agent_task"]
                and not state["active_agent_task"].done()
            ):
                state["active_agent_task"].cancel()
            state["active_agent_task"] = asyncio.create_task(
                run_agent_loop(user_text)
            )
            return None

        # ------------------------------------------------------------------
        # Senior Brain — main agent loop with swarm integration
        # ------------------------------------------------------------------

        async def run_agent_loop(user_text: str) -> None:  # noqa: C901
            logger.info(f"Agent loop starting: {user_text}")
            ctx = _app_ctx_summary()
            leetcode_mode = is_leetcode_task(user_text)
            step_limit = LEETCODE_MAX_AGENT_STEPS if leetcode_mode else MAX_AGENT_STEPS
            extra_guidance = task_specific_guidance(user_text)

            # Step 0: Memory sub-agent injects relevant past context
            memory = await memory_sub_agent(user_text)
            memory_block = ""
            if memory.relevant_facts or memory.session_summary:
                parts: list[str] = []
                if memory.session_summary:
                    parts.append(f"Past context: {memory.session_summary}")
                if memory.relevant_facts:
                    parts.append("Relevant facts: " + "; ".join(memory.relevant_facts))
                memory_block = "\n\nMEMORY:\n" + "\n".join(parts)

            SYSTEM_PROMPT = f"""You are the Senior Vision Specialist for Sai, a macOS desktop agent.

SCREEN CONTEXT: {ctx}{memory_block}

STRATEGIC APPROACH:
1. On your FIRST step, analyze the screenshot and form a numbered PLAN in the "explanation" field.
2. Every subsequent "explanation" must say WHY the action advances the plan — not just what you clicked.
3. If an action doesn't clearly move toward the goal, do NOT take it.

RULES:
- If the relevant app/website is ALREADY open, work within it. Never open Spotlight or navigate away unless necessary.
- If the user asked to open a browser/site and it is not open yet, prefer open_app/open_url before clicking around the current app.
- You can READ all text visible in the screenshot. Do NOT click elements just to read their content.
- Use keyboard_type for ALL on-screen text entry. type_text is ONLY for Spotlight app launching.
- NEVER use paste hotkeys (Command+V / Cmd+V).
- ONE action per step. After each action, verify the result in the next screenshot.
- Set done=true only when the screenshot CONFIRMS the task is fully complete.
{extra_guidance if extra_guidance else ""}

COORDINATE SYSTEM: Normalized [0, 1000] x [0, 1000]. (0,0)=top-left, (1000,1000)=bottom-right.
The ruler tick marks on the screenshot edges are your spatial reference.

Output ONLY valid JSON:
{{"explanation":"...","command":"click|type_text|open_app|keyboard_type|open_url|press_hotkey|scroll|wait","x":0-1000,"y":0-1000,"text":"...","app":"...","url":"...","browser":"...","keys":[...],"amount":0,"done":false}}"""

            conversation_history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            last_action: typing.Optional[dict] = None
            recent_actions: list[str] = []
            task_outcome = "incomplete"

            def _sig(a: AgentAction) -> str:
                if a.command == "click":
                    return f"click({a.x},{a.y})"
                if a.command in ("keyboard_type", "type_text"):
                    return f"{a.command}({(a.text or '')[:30]})"
                if a.command == "scroll":
                    return f"scroll({a.amount})"
                if a.command == "press_hotkey":
                    return f"hotkey({a.keys})"
                if a.command == "open_url":
                    return f"url({(a.url or '')[:40]})"
                if a.command == "open_app":
                    return f"app({(a.app or '')[:40]})"
                return a.command

            def _detect_cycle(actions: list[str]) -> typing.Optional[int]:
                n = len(actions)
                if n < 4:
                    return None
                for length in range(1, min(n // 2 + 1, 7)):
                    if actions[-length:] == actions[-length * 2 : -length]:
                        return length
                return None

            def _parse_action(raw: str) -> AgentAction:
                """Extract JSON from raw model output and validate with Pydantic."""
                clean = raw.strip()
                if clean.startswith("```"):
                    clean = "\n".join(
                        line for line in clean.splitlines()
                        if not line.startswith("```")
                    ).strip()
                if not clean.startswith("{"):
                    start, end = clean.find("{"), clean.rfind("}")
                    if start != -1 and end != -1:
                        clean = clean[start : end + 1]
                return AgentAction.model_validate_json(clean)

            try:
                # Wait for initial screenshot
                if not state["latest_screenshot_b64"]:
                    state["screenshot_event"].clear()
                    await websocket.send_text(json.dumps({"command": "capture_screen"}))
                    try:
                        await asyncio.wait_for(
                            state["screenshot_event"].wait(), timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.error("Initial screenshot timed out.")
                        return

                for step in range(step_limit):
                    screenshot = state["latest_screenshot_b64"]
                    if not screenshot:
                        logger.error("No screenshot available at step start.")
                        break

                    logger.info(f"Step {step + 1}/{step_limit}")
                    annotated = annotate_screenshot(screenshot, last_action)

                    # Build user message for this step
                    if step == 0:
                        user_msg = (
                            f"Task: {user_text}\n"
                            "Analyze the screenshot carefully. In 'explanation', describe your HIGH-LEVEL PLAN "
                            "(numbered steps), then take the FIRST action.\n"
                            "Remember: you can READ all visible text — do NOT click just to read."
                        )
                    else:
                        user_msg = (
                            f"Task (reminder): {user_text}\n"
                            "Evaluate whether your last action succeeded, then take the next step."
                        )

                    if leetcode_mode:
                        user_msg += (
                            "\nLeetCode reminder: do not finish until the page visibly shows Accepted or an "
                            "equivalent successful submission. If the latest result shows any failure, fix the "
                            "code and submit again."
                        )

                    # Stuck detection
                    cycle = _detect_cycle(recent_actions)
                    if cycle is not None:
                        user_msg += (
                            f"\n\nCRITICAL: You are stuck in a loop of {cycle} repeating actions: "
                            f"{recent_actions[-cycle:]}. "
                            "MUST completely change approach. Ask: What is the ACTUAL GOAL? "
                            "Should I be TYPING instead of clicking? Take a fundamentally different action NOW."
                        )
                    elif len(recent_actions) >= 2 and len(set(recent_actions[-2:])) == 1:
                        user_msg += (
                            f"\n\nWARNING: Same action repeated twice ({recent_actions[-1]}). "
                            "It is not working. Try a completely different approach."
                        )

                    conversation_history.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_msg},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{annotated}"},
                            },
                        ],
                    })

                    # Compress history to prevent context overflow
                    conversation_history = await _compress_history(
                        conversation_history, user_text
                    )

                    # --- Senior Brain call (blocking sync → executor) ---
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: nova_pro_client.chat.completions.create(
                            model=NOVA_PRO_MODEL_ID,
                            messages=conversation_history,
                            temperature=0.2,
                        ),
                    )
                    raw_content: str = response.choices[0].message.content
                    logger.info(f"Senior Brain: {raw_content[:200]}...")
                    conversation_history.append(
                        {"role": "assistant", "content": raw_content}
                    )

                    # --- Parse into Pydantic model ---
                    try:
                        action = _parse_action(raw_content)
                    except Exception as exc:
                        logger.error(f"AgentAction parse failed: {exc}")
                        conversation_history.append({
                            "role": "user",
                            "content": (
                                f"Your response was not valid JSON. Error: {exc}. "
                                "Output ONLY a valid JSON object matching the schema. No text outside the JSON."
                            ),
                        })
                        continue

                    # --- Coordinate guardrail ---
                    if not _coords_valid(action):
                        logger.warning(
                            f"Coordinate out of bounds: ({action.x}, {action.y}). Injecting correction."
                        )
                        conversation_history.append({
                            "role": "user",
                            "content": (
                                f"GUARDRAIL: click at ({action.x}, {action.y}) is outside [0, 1000]. "
                                "All coordinates MUST be in range [0, 1000]. "
                                "Re-examine the screenshot and output a corrected action."
                            ),
                        })
                        continue

                    # --- Critic sub-agent (click verification) ---
                    if action.command == "click":
                        verdict = await critic_sub_agent(action, annotated)
                        if not verdict.approved:
                            logger.warning(
                                f"Critic rejected click ({action.x},{action.y}): {verdict.reason}"
                            )
                            if (
                                verdict.corrected_x is not None
                                and verdict.corrected_y is not None
                            ):
                                action = action.model_copy(
                                    update={
                                        "x": verdict.corrected_x,
                                        "y": verdict.corrected_y,
                                    }
                                )
                                logger.info(
                                    f"Critic corrected click → ({action.x},{action.y})"
                                )
                            else:
                                conversation_history.append({
                                    "role": "user",
                                    "content": (
                                        f"CRITIC: The planned click at ({action.x},{action.y}) does not land "
                                        f"on an interactive element. Reason: {verdict.reason}. "
                                        "Re-examine the screenshot and choose a different target."
                                    ),
                                })
                                continue

                    # --- Verification gate: forbid done=true on action steps ---
                    if action.done and action.command not in ("wait",):
                        logger.warning(
                            f"Agent tried to signal done=true on an action step ({action.command}). "
                            "Forcing done=false — must wait for next screenshot to confirm."
                        )
                        action = action.model_copy(update={"done": False})

                    # --- Execute command ---
                    cmd = action.command
                    if cmd in {
                        "open_url", "open_app", "click", "type_text", "keyboard_type",
                        "press_hotkey", "scroll", "wait",
                    }:
                        payload = action.model_dump(exclude_none=True)
                        last_action = payload
                        recent_actions.append(_sig(action))
                        logger.info(f"Executing: {cmd}")
                        await websocket.send_text(json.dumps(payload))
                        settle_time = ACTION_SETTLE_TIME
                        if leetcode_mode and cmd in {"click", "keyboard_type", "press_hotkey", "wait", "open_url", "open_app"}:
                            settle_time = max(settle_time, 3.0)
                        await asyncio.sleep(settle_time)

                        # Hard cycle bail (3 full repetitions after warnings)
                        if len(recent_actions) >= 6:
                            cyc = _detect_cycle(recent_actions)
                            if cyc is not None:
                                total = cyc * 3
                                if len(recent_actions) >= total:
                                    tail = recent_actions[-total:]
                                    chunks = [
                                        tuple(tail[i * cyc : (i + 1) * cyc])
                                        for i in range(3)
                                    ]
                                    if len(set(chunks)) == 1:
                                        logger.error(
                                            f"Hopelessly stuck in cycle of {cyc}, aborting."
                                        )
                                        break

                    # --- Check for completion after action execution ---
                    if action.done:
                        logger.info("Agent signaled done=true.")
                        task_outcome = "accepted" if leetcode_mode else "complete"
                        await send_voice_reply(f"Task completed: {user_text}")
                        break

                    # --- Request next screenshot ---
                    state["screenshot_event"].clear()
                    await websocket.send_text(json.dumps({"command": "capture_screen"}))
                    try:
                        await asyncio.wait_for(
                            state["screenshot_event"].wait(),
                            timeout=SCREENSHOT_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.error("Screenshot timed out in agent loop.")
                        break

            except asyncio.CancelledError:
                logger.info("Agent loop cancelled.")
            except Exception as exc:
                import traceback
                error_text = str(exc)
                if "401" in error_text or "User not found" in error_text:
                    task_outcome = "failed_openrouter_auth"
                    logger.error(
                        "OpenRouter auth failed. Use a regular inference API key, "
                        "not a management/provisioning key."
                    )
                    await send_voice_reply(
                        "The vision agent cannot run yet because the OpenRouter key "
                        "cannot call chat completions. Please create a regular "
                        "OpenRouter API key and restart the server."
                    )
                    return
                logger.error(f"Agent loop error: {type(exc).__name__}: {exc}")
                logger.error(traceback.format_exc())
            finally:
                # Persist session outcome to memory DB
                try:
                    summary = (
                        f"Task: {user_text} | Steps: {len(recent_actions)} | "
                        f"Outcome: {task_outcome} | Last: {recent_actions[-3:]}"
                    )
                    store_session(user_text, task_outcome, summary)
                    logger.info(f"Session stored: {task_outcome}")
                except Exception as exc:
                    logger.warning(f"Failed to store session: {exc}")

        # ------------------------------------------------------------------
        # Transcription processing
        # ------------------------------------------------------------------

        async def process_complete_transcription() -> None:
            if state["command_triggered"]:
                return
            full_text = " ".join(state["transcription_buffer"]).strip()
            state["transcription_buffer"] = []
            if not full_text:
                return

            state["command_triggered"] = True
            logger.info(f"Raw transcription: {full_text}")
            full_text = strip_wake_phrase(full_text)
            logger.info(f"Command after wake phrase cleanup: {full_text}")

            if is_stop_command(full_text):
                logger.info("Stop command received. Asking client to shut down.")
                await send_voice_reply("Stopping Sai.")
                await websocket.send_text(json.dumps({"command": "shutdown"}))
                await websocket.close()
                return

            if state["debounce_task"]:
                state["debounce_task"].cancel()

            try:
                await websocket.send_text(
                    json.dumps({"command": "set_activity", "state": "active"})
                )
            except Exception as exc:
                logger.warning(f"Failed to send activity start: {exc}")

            try:
                skip_done_reply = False
                if start_actions := deterministic_start_actions(full_text):
                    logger.info(f"Deterministic startup actions: {start_actions}")
                    for action in start_actions:
                        await websocket.send_text(json.dumps(action))
                        await asyncio.sleep(ACTION_SETTLE_TIME)
                    if not should_continue_after_start_actions(full_text):
                        return
                if action := deterministic_simple_action(full_text):
                    logger.info(f"Raw deterministic SIMPLE action: {action}")
                    await websocket.send_text(json.dumps(action))
                    await asyncio.sleep(ACTION_SETTLE_TIME)
                    return
                interpreted = await interpret_intent(full_text)
                logger.info(f"Processing: {interpreted}")
                if start_actions := deterministic_start_actions(interpreted):
                    logger.info(f"Deterministic startup actions: {start_actions}")
                    for action in start_actions:
                        await websocket.send_text(json.dumps(action))
                        await asyncio.sleep(ACTION_SETTLE_TIME)
                    if not should_continue_after_start_actions(interpreted):
                        return
                if action := deterministic_simple_action(interpreted):
                    logger.info(f"Deterministic SIMPLE action: {action}")
                    await websocket.send_text(json.dumps(action))
                    await asyncio.sleep(ACTION_SETTLE_TIME)
                    return
                if response := local_simple_response(interpreted):
                    await send_voice_reply(response)
                    skip_done_reply = True
                    return
                await send_voice_reply("Got it. Working on that.")
                action = await hybrid_reasoning(interpreted)
                if action:
                    await websocket.send_text(json.dumps(action))
                    await asyncio.sleep(ACTION_SETTLE_TIME)
                    return
                if state["active_agent_task"]:
                    try:
                        await state["active_agent_task"]
                    except asyncio.CancelledError:
                        pass
            finally:
                if not locals().get("skip_done_reply", False):
                    await send_voice_reply("Done.")
                try:
                    await websocket.send_text(
                        json.dumps({"command": "set_activity", "state": "idle"})
                    )
                except Exception as exc:
                    logger.warning(f"Failed to send activity stop: {exc}")
                logger.info("Command complete. Closing session.")
                await websocket.close()

        async def handle_client_event(data: dict) -> None:
            if data.get("event") == "screen_captured":
                state["latest_screenshot_b64"] = data.get("image_base64")
                state["latest_app_context"] = data.get("app_context", {})
                # Record physical screen dimensions on first capture
                if state["screen_width"] is None:
                    state["screen_width"] = data.get("width")
                    state["screen_height"] = data.get("height")
                    logger.info(
                        f"Screen dimensions: {state['screen_width']}x{state['screen_height']}"
                    )
                state["screenshot_event"].set()
            elif data.get("event") == "screen_capture_failed":
                logger.error(
                    "Client screen capture failed: %s",
                    data.get("error", "unknown error"),
                )

        if state["manual_text_command"]:
            logger.info("Processing manual text command from client.")
            state["transcription_buffer"] = [state["manual_text_command"]]
            await process_complete_transcription()
            return

        async def run_deepgram_stt_loop() -> None:
            dg_url = (
                "wss://api.deepgram.com/v1/listen"
                f"?model={DEEPGRAM_STT_MODEL}"
                "&encoding=linear16"
                f"&sample_rate={STT_SAMPLE_RATE}"
                "&channels=1"
                "&language=en"
                "&interim_results=true"
                "&smart_format=true"
                "&endpointing=1000"
                "&utterance_end_ms=1000"
                "&vad_events=true"
            )
            async with ws_client.connect(
                dg_url,
                additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            ) as dg_ws:
                logger.info(
                    "Connected to Deepgram STT: model=%s sample_rate=%s",
                    DEEPGRAM_STT_MODEL,
                    STT_SAMPLE_RATE,
                )

                async def listen_deepgram() -> None:
                    final_parts: list[str] = []
                    finalize_task: typing.Optional[asyncio.Task] = None

                    async def commit_final_parts(reason: str) -> None:
                        nonlocal final_parts, finalize_task
                        if finalize_task and not finalize_task.done():
                            finalize_task.cancel()
                        finalize_task = None
                        text = " ".join(final_parts).strip()
                        final_parts = []
                        if text:
                            logger.info("Deepgram %s: %s", reason, text)
                            state["transcription_buffer"].append(text)
                            await process_complete_transcription()

                    async def delayed_commit() -> None:
                        try:
                            await asyncio.sleep(1.2)
                            await commit_final_parts("final after silence")
                        except asyncio.CancelledError:
                            pass

                    try:
                        async for msg in dg_ws:
                            if state["command_triggered"]:
                                continue
                            if isinstance(msg, bytes):
                                continue
                            try:
                                data = json.loads(msg)
                            except json.JSONDecodeError:
                                logger.warning("Non-JSON from Deepgram: %s", str(msg)[:100])
                                continue

                            msg_type = data.get("type", "")
                            if msg_type == "Metadata":
                                logger.info("Deepgram metadata received")
                                continue
                            if msg_type == "SpeechStarted":
                                logger.info("Deepgram speech started")
                                continue
                            if msg_type not in {"Results", "UtteranceEnd"}:
                                if msg_type in {"Error", "Warning"}:
                                    logger.error("Deepgram %s: %s", msg_type, data)
                                continue

                            if msg_type == "UtteranceEnd":
                                await commit_final_parts("utterance")
                                continue

                            channel = data.get("channel", {})
                            alternatives = channel.get("alternatives", [])
                            transcript = ""
                            if alternatives:
                                transcript = alternatives[0].get("transcript", "").strip()
                            if not transcript:
                                continue

                            if data.get("is_final"):
                                logger.info("Deepgram final: %s", transcript)
                                final_parts.append(transcript)
                                if finalize_task and not finalize_task.done():
                                    finalize_task.cancel()
                                finalize_task = asyncio.create_task(delayed_commit())
                            else:
                                logger.info("Deepgram partial: %s", transcript)

                            if data.get("speech_final"):
                                if not final_parts and transcript:
                                    final_parts.append(transcript)
                                await commit_final_parts("speech final")
                    except Exception as exc:
                        logger.error("Deepgram listener error: %s", exc)
                    finally:
                        if finalize_task and not finalize_task.done():
                            finalize_task.cancel()

                dg_listen_task = asyncio.create_task(listen_deepgram())

                try:
                    while True:
                        try:
                            message = await websocket.receive()
                        except RuntimeError:
                            break

                        if message.get("bytes"):
                            if not state["command_triggered"]:
                                try:
                                    await dg_ws.send(message["bytes"])
                                except Exception as exc:
                                    logger.error("Failed to forward audio to Deepgram: %s", exc)
                        elif message.get("text"):
                            try:
                                await handle_client_event(json.loads(message["text"]))
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Non-JSON text message from client: %s",
                                    message["text"][:100],
                                )
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                finally:
                    dg_listen_task.cancel()

        if STT_PROVIDER == "deepgram":
            await run_deepgram_stt_loop()
            return

        async def run_gemini_stt_loop() -> None:
            logger.info(
                "Using Gemini STT: model=%s rms_threshold=%s silence=%.1fs",
                GEMINI_STT_MODEL,
                STT_RMS_THRESHOLD,
                STT_SILENCE_SECS,
            )
            audio_buffer = bytearray()
            pre_roll: list[bytes] = []
            pre_roll_bytes = int(STT_SAMPLE_RATE * 2 * 0.35)
            speech_started = False
            speech_secs = 0.0
            last_voice_time = time.monotonic()
            utterance_start_time = time.monotonic()

            async def finalize_utterance() -> None:
                nonlocal audio_buffer, pre_roll, speech_started, speech_secs
                nonlocal last_voice_time, utterance_start_time

                if state["command_triggered"]:
                    return

                pcm = bytes(audio_buffer)
                audio_buffer = bytearray()
                pre_roll = []
                speech_started = False
                last_voice_time = time.monotonic()
                utterance_start_time = time.monotonic()

                if speech_secs < STT_MIN_SPEECH_SECS:
                    logger.info(
                        "Ignoring short speech segment: %.2fs < %.2fs",
                        speech_secs,
                        STT_MIN_SPEECH_SECS,
                    )
                    speech_secs = 0.0
                    return

                speech_secs = 0.0
                try:
                    loop = asyncio.get_running_loop()
                    text = await loop.run_in_executor(
                        None,
                        transcribe_speech_with_gemini,
                        pcm,
                    )
                    if text:
                        logger.info("Gemini transcript: %s", text)
                        state["transcription_buffer"].append(text)
                        await process_complete_transcription()
                    else:
                        logger.info("Gemini transcript was empty.")
                except Exception as exc:
                    logger.error("Gemini STT error: %s", exc)

            while True:
                try:
                    message = await websocket.receive()
                except RuntimeError:
                    break

                if message.get("bytes"):
                    if state["command_triggered"]:
                        continue

                    pcm = message["bytes"]
                    chunk_secs = len(pcm) / 2 / STT_SAMPLE_RATE
                    rms = _pcm16_rms(pcm)
                    now = time.monotonic()

                    if not speech_started and rms < STT_RMS_THRESHOLD:
                        pre_roll.append(pcm)
                        while sum(len(chunk) for chunk in pre_roll) > pre_roll_bytes:
                            pre_roll.pop(0)

                    if rms >= STT_RMS_THRESHOLD:
                        if not speech_started:
                            speech_started = True
                            utterance_start_time = now
                            audio_buffer.extend(b"".join(pre_roll))
                            pre_roll = []
                            logger.info("Speech detected by Gemini VAD: rms=%.0f", rms)
                        audio_buffer.extend(pcm)
                        speech_secs += chunk_secs
                        last_voice_time = now
                    elif speech_started:
                        audio_buffer.extend(pcm)

                    if speech_started and (
                        now - last_voice_time >= STT_SILENCE_SECS
                        or now - utterance_start_time >= STT_MAX_UTTERANCE_SECS
                    ):
                        await finalize_utterance()

                elif message.get("text"):
                    try:
                        await handle_client_event(json.loads(message["text"]))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Non-JSON text message from client: %s",
                            message["text"][:100],
                        )

        if STT_PROVIDER == "gemini":
            await run_gemini_stt_loop()
            return

        # ------------------------------------------------------------------
        # ElevenLabs STT connection
        # ------------------------------------------------------------------

        el_url = (
            ELEVENLABS_STT_WS_URL
            + "?model_id=scribe_v2_realtime"
            + "&language_code=en"
            + "&audio_format=pcm_16000"
            + "&commit_strategy=vad"
            + "&vad_silence_threshold_secs=1.2"
        )

        async with ws_client.connect(
            el_url, additional_headers={"xi-api-key": ELEVENLABS_API_KEY}
        ) as el_ws:
            logger.info("Connected to ElevenLabs Scribe v2 Realtime STT")

            async def listen_elevenlabs() -> None:
                try:
                    async for msg in el_ws:
                        if state["command_triggered"]:
                            continue
                        try:
                            data = json.loads(msg)
                            mtype = data.get("message_type", "")
                            if mtype == "session_started":
                                logger.info(f"ElevenLabs session: {data.get('session_id')}")
                            elif mtype == "partial_transcript":
                                if text := data.get("text", "").strip():
                                    logger.info(f"Partial: {text}")
                            elif mtype in (
                                "committed_transcript",
                                "committed_transcript_with_timestamps",
                            ):
                                if text := data.get("text", "").strip():
                                    logger.info(f"Committed: {text}")
                                    state["transcription_buffer"].append(text)
                                    if state["debounce_task"]:
                                        state["debounce_task"].cancel()
                                    await process_complete_transcription()
                            elif mtype in (
                                "error", "auth_error", "quota_exceeded",
                                "rate_limited", "transcriber_error",
                            ):
                                logger.error(f"ElevenLabs error: {data}")
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON from ElevenLabs: {str(msg)[:100]}")
                except Exception as exc:
                    logger.error(f"ElevenLabs listener error: {exc}")

            el_listen_task = asyncio.create_task(listen_elevenlabs())

            try:
                while True:
                    try:
                        message = await websocket.receive()
                    except RuntimeError:
                        break

                    if message.get("bytes"):
                        if not state["command_triggered"]:
                            pcm = message["bytes"]
                            chunk = json.dumps({
                                "message_type": "input_audio_chunk",
                                "audio_base_64": base64.b64encode(pcm).decode("utf-8"),
                                "commit": False,
                                "sample_rate": 16000,
                            })
                            try:
                                await el_ws.send(chunk)
                            except Exception as exc:
                                logger.error(f"Failed to forward audio: {exc}")

                    elif message.get("text"):
                        try:
                            await handle_client_event(json.loads(message["text"]))
                        except json.JSONDecodeError:
                            logger.warning(
                                "Non-JSON text message from client: %s",
                                message["text"][:100],
                            )

            except WebSocketDisconnect:
                logger.info("Client disconnected")
            finally:
                el_listen_task.cancel()

    except Exception as exc:
        import traceback
        logger.error(f"WebSocket handler error: {exc}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Session ended")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
