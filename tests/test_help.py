"""Every control in the interface must say what it is for.

A tooltip is the cheapest documentation there is: it is where somebody looks
when they are already in front of the thing they do not understand.  This test
fails when a new control arrives without one.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

INTERACTIVE = {"input", "select", "button", "textarea", "a"}
#: Containers that can carry the help for the controls inside them.
CONTAINERS = {"label", "span", "div", "details", "fieldset", "section"}
INDEX = Path(__file__).resolve().parents[1] / "ptmipf" / "webui" / "static" / "index.html"


class _Help(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack: list[str | None] = []
        self.missing: list[tuple[int, str, str]] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in CONTAINERS:
            self.stack.append(attributes.get("title"))
        if tag in INTERACTIVE and attributes.get("type") != "hidden":
            if not (attributes.get("title") or any(self.stack)):
                name = attributes.get("id") or attributes.get("class") or tag
                self.missing.append((self.getpos()[0], tag, name))

    def handle_endtag(self, tag):
        if tag in CONTAINERS and self.stack:
            self.stack.pop()


def test_every_control_has_help_text():
    parser = _Help()
    parser.feed(INDEX.read_text(encoding="utf-8"))
    assert parser.missing == [], (
        "these controls have no title of their own and none on a container around "
        f"them: {parser.missing}"
    )


def test_the_help_is_a_sentence_not_a_word():
    """A title of one word repeats the label and helps nobody."""
    parser = _Help()
    text = INDEX.read_text(encoding="utf-8")
    parser.feed(text)
    short = [
        title
        for title in _titles(text)
        if len(title.split()) < 2
    ]
    assert short == [], f"these titles say too little: {short}"


def _titles(text: str) -> list[str]:
    import re

    return [match.group(1) for match in re.finditer(r'title="([^"]*)"', text)]
