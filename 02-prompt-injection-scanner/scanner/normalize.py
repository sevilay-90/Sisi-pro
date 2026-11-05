"""Text normalization that undoes common obfuscation tricks before scanning.

Attackers hide instructions with invisible characters, look-alike letters,
spacing ("i g n o r e") or encodings (base64). We expose a cleaned view of the
text and report which tricks were found, since the tricks themselves are a signal.
"""

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field

# Zero-width and other invisible format characters, plus Unicode "tag" characters
# (U+E0000-U+E007F) which can smuggle hidden ASCII text.
INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿\U000E0000-\U000E007F]")

# A few frequent Cyrillic/Greek look-alikes of Latin letters.
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Т": "T", "Н": "H", "К": "K",
    "І": "I", "В": "B", "М": "M", "Х": "X", "ј": "j", "ѕ": "s",
    "ο": "o", "α": "a", "ε": "e", "ι": "i",
})

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

SPACED_WORD = re.compile(r"\b(?:[A-Za-z][ .\-_]){3,}[A-Za-z]\b")
BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


@dataclass
class Normalized:
    original: str
    text: str                       # cleaned text used for matching
    decoded: list = field(default_factory=list)   # decoded hidden payloads
    tricks: list = field(default_factory=list)    # obfuscation techniques detected


def _decode_tags(text: str) -> str:
    """Unicode tag characters map 1:1 onto ASCII; recover any hidden message."""
    hidden = "".join(chr(ord(c) - 0xE0000) for c in text if 0xE0020 <= ord(c) <= 0xE007E)
    return hidden


def normalize(text: str) -> Normalized:
    tricks, decoded = [], []

    hidden = _decode_tags(text)
    if hidden.strip():
        decoded.append(hidden)
        tricks.append("unicode_tag_smuggling")

    if INVISIBLE.search(text):
        tricks.append("invisible_characters")
    clean = INVISIBLE.sub("", text)

    nfkc = unicodedata.normalize("NFKC", clean)
    if nfkc != clean:
        tricks.append("unicode_compatibility_forms")
    clean = nfkc

    unglyphed = clean.translate(HOMOGLYPHS)
    if unglyphed != clean:
        tricks.append("homoglyphs")
    clean = unglyphed

    def _join(m):
        return re.sub(r"[ .\-_]", "", m.group(0))
    joined = SPACED_WORD.sub(_join, clean)
    if joined != clean:
        tricks.append("letter_spacing")
    clean = joined

    for blob in BASE64_BLOB.findall(text):
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
            msg = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if sum(ch.isprintable() for ch in msg) / max(len(msg), 1) > 0.9 and " " in msg:
            decoded.append(msg)
            if "base64_payload" not in tricks:
                tricks.append("base64_payload")

    return Normalized(original=text, text=clean, decoded=decoded, tricks=tricks)


def leet_view(text: str) -> str:
    """Leetspeak-folded copy ("1gn0r3" -> "ignore"), used as an extra match pass."""
    return text.lower().translate(LEET)
