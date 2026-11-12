"""Turning assistant text into something worth reading aloud.

The agent is asked to reply in plain sentences (see VOICE_PROMPT in
server.py), but markdown still slips through, and Piper reads every symbol
literally ("asterisk asterisk"). to_speakable() runs on each assistant text
block before it's spoken and shown as the subtitle.

To silence or rename a symbol, edit the tables at the top. The steps run in
the order listed in to_speakable().
"""

import re

# Fenced code blocks are never read aloud — this is said in their place.
CODE_BLOCK_REPLACEMENT = "I've left out a code block."

# Bare URLs are replaced with this. Markdown links keep their label instead.
URL_REPLACEMENT = "a link"

# Symbols that should be spoken as a word. Matched with spaces around them,
# so "a -> b" becomes "a to b" but "foo-bar" is left alone.
SPOKEN_SYMBOLS = {
    "->": "to",
    "=>": "to",
    "→": "to",
    "&": "and",
}

# Characters that are only formatting and are removed outright, wherever
# they appear. Underscores become spaces separately (see below) so
# snake_case words still read as words.
SILENT_CHARS = "*`~#>|"

_CODE_FENCE = re.compile(r"```.*?(```|$)", re.DOTALL)
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_HEADING_OR_QUOTE = re.compile(r"^\s*(?:#{1,6}|>+)\s*")
_HORIZONTAL_RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
_SENTENCE_END = re.compile(r"[.!?:;,]$")


def _clean_line(line: str) -> str:
    if _TABLE_SEPARATOR.match(line) or _HORIZONTAL_RULE.match(line):
        return ""
    line = _HEADING_OR_QUOTE.sub("", line)
    line = _LIST_MARKER.sub("", line)
    # Table rows: "| a | b |" reads as "a, b".
    if line.count("|") >= 2:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        line = ", ".join(c for c in cells if c)
    return line.strip()


def to_speakable(text: str) -> str:
    """Returns `text` with markdown and symbols removed, as plain sentences.
    Returns "" if nothing speakable is left."""
    text = _CODE_FENCE.sub(f"\n{CODE_BLOCK_REPLACEMENT}\n", text)
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(URL_REPLACEMENT, text)

    # Each line becomes its own sentence, so list items and headings get a
    # pause instead of running together.
    sentences = []
    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue
        if not _SENTENCE_END.search(line):
            line += "."
        sentences.append(line)
    text = " ".join(sentences)

    for symbol, word in SPOKEN_SYMBOLS.items():
        text = text.replace(f" {symbol} ", f" {word} ")
    text = text.translate(str.maketrans("", "", SILENT_CHARS))
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    # A line that was only symbols leaves a lone "." behind.
    text = re.sub(r"(^|\s)[.]+(?=\s|$)", "", text).strip()
    return text
