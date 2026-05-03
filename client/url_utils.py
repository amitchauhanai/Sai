import re
from urllib.parse import quote_plus

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

DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?::\d+)?(?:[/?#][^\s]*)?$",
    re.IGNORECASE,
)
SINGLE_HOST_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$", re.IGNORECASE)
OPEN_IN_TAB_RE = re.compile(
    r"\b(?:in|into|on|with)\s+(?:a\s+|the\s+)?(?:new\s+)?(?:browser\s+)?tab\b",
    re.IGNORECASE,
)
TAB_TARGET_PREFIX_RE = re.compile(
    r"^(?:a\s+|the\s+)?(?:new\s+)?(?:browser\s+)?tab\s+(?:for|to|with)\s+",
    re.IGNORECASE,
)


def _clean_target(raw: str) -> str:
    target = (raw or "").strip().strip("\"'")
    target = TAB_TARGET_PREFIX_RE.sub("", target)
    target = OPEN_IN_TAB_RE.sub("", target)
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


def normalize_url_for_open(raw: str) -> str:
    """
    Convert casual spoken website targets into something macOS can open.

    Examples:
    - "google" -> "https://www.google.com"
    - "github dot com" -> "https://github.com"
    - "weather in delhi" -> Google search URL
    """
    target = _clean_target(raw)
    if not target:
        return ""

    lowered = target.lower()
    if lowered in KNOWN_SITE_ALIASES:
        return KNOWN_SITE_ALIASES[lowered]

    urlish = _spoken_target_to_urlish(target)
    compact = re.sub(r"\s+", "", urlish)

    if compact in KNOWN_SITE_ALIASES:
        return KNOWN_SITE_ALIASES[compact]

    if DOMAIN_RE.fullmatch(compact):
        if compact.startswith(("http://", "https://")):
            return compact
        return f"https://{compact}"

    if " " in target:
        return f"https://www.google.com/search?q={quote_plus(target)}"

    if SINGLE_HOST_RE.fullmatch(compact):
        return f"https://www.{compact}.com"

    return target
