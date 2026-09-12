"""Extraction de texte depuis du HTML via le parseur stdlib.

Ce module ne rend pas le HTML « sûr ». Il convertit du HTML en texte,
ce qui est le bon outil quand le contrat métier est du texte brut.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_SKIP_TAGS = frozenset({"script", "style"})
_BREAK_TAGS = frozenset(
    {"br", "p", "div", "li", "h1", "h2", "h3", "tr", "table", "blockquote"}
)


class _HtmlTextParser(HTMLParser):
    def __init__(self, *, keep_block_breaks: bool) -> None:
        super().__init__(convert_charrefs=True)
        self._keep_block_breaks = keep_block_breaks
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if self._keep_block_breaks and name in _BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in _SKIP_TAGS and self._skip:
            self._skip -= 1
            return
        if self._skip:
            return
        if self._keep_block_breaks and name in _BREAK_TAGS and name != "br":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def contains_html_tag(value: str) -> bool:
    """True s'il existe une sous-chaîne ``<...>`` (au moins un caractère entre).

    Même contrat que l'ancien ``<[^>]+>`` : « 1 < 2 » et « Jean > Paul »
    restent du texte. Ce n'est pas un sanitizer XSS.
    Parcours linéaire, sans backtracking.
    """
    if not value:
        return False
    start = 0
    while True:
        open_at = value.find("<", start)
        if open_at == -1:
            return False
        close_at = value.find(">", open_at + 1)
        if close_at == -1:
            return False
        if close_at > open_at + 1:
            return True
        start = open_at + 1


def strip_html_to_text(value: str) -> str:
    """Retire le balisage et ignore le contenu script/style."""
    parser = _HtmlTextParser(keep_block_breaks=False)
    parser.feed(value)
    parser.close()
    return "".join(parser.parts)


def html_to_plain_text(html_content: str) -> str:
    """Convertit un HTML simple en texte brut (alternative MIME)."""
    parser = _HtmlTextParser(keep_block_breaks=True)
    parser.feed(html_content)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
