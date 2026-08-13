from __future__ import annotations

import re
import string


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
