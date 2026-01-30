from __future__ import annotations

import os
import re
import unicodedata
from typing import List, Set, Tuple, Literal, Optional

_PUNCT_TO_SPACE_RE = re.compile(r"[^0-9a-zA-Z']+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")

Lang = Literal["en", "zh", "mixed"]


def _nfkc_lower(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    return s.lower()


def _is_cjk(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    # CJK Unified Ideographs + Extension A (good enough heuristic for Chinese/mixed CJK).
    return (0x3400 <= cp <= 0x4DBF) or (0x4E00 <= cp <= 0x9FFF)


def detect_lang(text: str) -> Lang:
    """
    Heuristic language detection for routing:
    - zh: contains any CJK ideographs and no Latin letters
    - mixed: contains both CJK ideographs and Latin letters
    - en: otherwise
    """
    s = text or ""
    has_cjk = any(_is_cjk(ch) for ch in s)
    has_latin = any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in s)
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    return "en"


def normalize_text(text: str, lang: Optional[Lang] = None) -> str:
    """
    Unicode-safe normalization:
    - Always NFKC normalize
    - Remove zero-width characters
    - For zh/mixed: preserve CJK and punctuation (do not lowercase)
    - For en: preserve prior behavior (lowercase later in tokenizers)
    """
    s = unicodedata.normalize("NFKC", text or "")
    s = _ZERO_WIDTH_RE.sub("", s)
    # Keep caller control for casing; default routing uses lang when provided.
    _ = lang  # explicit for readability
    return s


def tokenize_en(text: str) -> List[str]:
    if not text:
        return []
    s = _nfkc_lower(text)
    s = _PUNCT_TO_SPACE_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    if not s:
        return []
    return s.split(" ")


_LATIN_TOKEN_RE = re.compile(r"[0-9a-zA-Z']+")


def tokenize_zh(text: str) -> List[str]:
    """
    Chinese/mixed tokenizer:
    - NFKC normalize, strip zero-width
    - Generate character bigrams for contiguous CJK segments
    - Preserve Latin tokens for mixed queries (lowercased)
    """
    s = normalize_text(text)
    if not s:
        return []

    # Collect CJK segments for bigrams.
    tokens: List[str] = []
    buf: List[str] = []
    for ch in s:
        if _is_cjk(ch):
            buf.append(ch)
            continue
        if buf:
            seg = "".join(buf)
            if len(seg) == 1:
                tokens.append(seg)
            else:
                tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
            buf.clear()
    if buf:
        seg = "".join(buf)
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))

    # Collect Latin tokens (casefolded) for mixed queries.
    for m in _LATIN_TOKEN_RE.finditer(s):
        tok = m.group(0).lower().strip()
        if tok:
            tokens.append(tok)

    return tokens

def tokenize(text: str) -> List[str]:
    """
    Deterministic tokenizer:
    - Unicode normalize (NFKC), lowercase
    - Replace punctuation with spaces (letters/digits and apostrophes kept)
    - Collapse whitespace
    - Split into tokens
    """
    # Backward-compatible default: English tokenization.
    return tokenize_en(text)


def get_stopwords() -> Set[str]:
    """
    Return English stopwords from NLTK. If resource is missing:
      - when CHATBOT_DEV_MODE=true, auto-download
      - else raise with remediation guidance
    """
    try:
        from nltk.corpus import stopwords  # type: ignore

        return set(w.lower() for w in stopwords.words("english"))
    except Exception:
        # Attempt auto-download when in dev mode
        dev_mode = str(os.getenv("CHATBOT_DEV_MODE", "false")).lower() not in {"0", "false", "no"}
        if dev_mode:
            try:
                import nltk  # type: ignore

                nltk.download("stopwords", quiet=True)
                from nltk.corpus import stopwords  # type: ignore

                return set(w.lower() for w in stopwords.words("english"))
            except Exception as e:
                raise RuntimeError(
                    f"NLTK stopwords resource missing and auto-download failed: {e}. "
                    "Run: python -c 'import nltk; nltk.download(\"stopwords\")'"
                )
        raise RuntimeError(
            "NLTK stopwords resource not available. "
            "Install nltk and run: python -c 'import nltk; nltk.download(\"stopwords\")' "
            "or set CHATBOT_DEV_MODE=true to auto-download at runtime."
        )


# Domain/entity routing noise (not general stopwords)
ENTITY_SUFFIXES: Set[str] = {
    "song",
    "track",
    "album",
    "artist",
    "band",
    "singer",
    "title",
    "named",
    "called",
    "titled",
}


def _remove_entity_suffix_tokens(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens
    return [t for t in tokens if t not in ENTITY_SUFFIXES]


def normalize_for_lexical(
    text: str,
    *,
    remove_stopwords: bool = True,
    remove_entity_suffixes: bool = True,
    max_tokens: int = 12,
    lang: Optional[Lang] = None,
) -> Tuple[str, List[str]]:
    """
    Produce a lexical query string and its tokens for FTS:
    - tokenize
    - guardrail: skip stopword removal when tokens <= 2
    - optionally remove entity suffix tokens
    - fallback to original tokens if empty after removals
    - cap tokens
    """
    lang_eff = lang or detect_lang(text)
    orig_tokens = tokenize_en(text) if lang_eff == "en" else tokenize_zh(text)
    tokens = list(orig_tokens)
    if remove_stopwords and len(tokens) > 2:
        try:
            sw = get_stopwords()
        except Exception:
            sw = set()  # fail open for production resilience
        if sw:
            # Stopwords are English-only; don't drop CJK bigrams.
            tokens = [t for t in tokens if (t not in sw)]
    if remove_entity_suffixes:
        tokens = _remove_entity_suffix_tokens(tokens)
    if not tokens:
        tokens = orig_tokens
    if not tokens:
        return ("", [])
    tokens = tokens[: max(1, int(max_tokens))]
    return (" ".join(tokens), tokens)


def normalize_for_fuzzy(text: str, *, remove_entity_suffixes: bool = True) -> str:
    """
    Normalization for fuzzy matching:
    - lowercase, punctuation stripped to spaces, whitespace collapsed
    - do NOT remove general stopwords
    - optionally remove entity suffix tokens if they appear as standalone tokens
    """
    lang_eff = detect_lang(text)
    if lang_eff == "en":
        tokens = tokenize_en(text)
        if remove_entity_suffixes:
            tokens = _remove_entity_suffix_tokens(tokens)
        if not tokens:
            return ""
        return " ".join(tokens)

    # zh/mixed: preserve CJK and Latin, normalize punctuation to spaces.
    s = normalize_text(text, lang=lang_eff)
    # Lowercase Latin only (helps mixed queries).
    s = "".join(ch.lower() if ("A" <= ch <= "Z") else ch for ch in s)
    s = re.sub(r"[^0-9a-zA-Z\u3400-\u9FFF']+", " ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    if not s:
        return ""
    if remove_entity_suffixes:
        # Only remove English entity suffixes when they appear as standalone ASCII tokens.
        parts = [p for p in s.split(" ") if p]
        parts = [p for p in parts if not (p.isascii() and p in ENTITY_SUFFIXES)]
        s = " ".join(parts).strip()
    return s

