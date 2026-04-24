import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)
TYPE_RE = re.compile(r"<TYPE>([^\n<]+)")
TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

HEADER_BLOCK_RE = re.compile(r"<SEC-HEADER>(.*?)</SEC-HEADER>", re.DOTALL)
ITEM_RE = re.compile(r"ITEM INFORMATION:\s*([^\n]+)")
COMPANY_NAME_RE = re.compile(r"COMPANY CONFORMED NAME:\s*([^\n]+)")
TICKER_RE = re.compile(r"(?:TRADING SYMBOL|Ticker Symbol)[:\s]+([A-Z\.\-]{1,8})")

ITEM_BODY_RE = re.compile(
    r"(?:Item\s+\d+\.\d+\.?\s+[A-Z][^\n]+)(.*?)(?=Item\s+\d+\.\d+|SIGNATURE|$)",
    re.DOTALL | re.IGNORECASE,
)


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self.chunks.append(data)

    def handle_starttag(self, tag, attrs) -> None:
        self.chunks.append(" ")

    def handle_endtag(self, tag) -> None:
        self.chunks.append(" ")

    def handle_startendtag(self, tag, attrs) -> None:
        self.chunks.append(" ")

    def text(self) -> str:
        return "".join(self.chunks)


def _html_to_text(html: str) -> str:
    p = _HTMLToText()
    try:
        p.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return p.text()


@dataclass
class ParsedFiling:
    form_type: str
    company_name: str = ""
    ticker: str = ""
    items: list[str] = field(default_factory=list)
    body_text: str = ""


def parse_filing(raw: str) -> ParsedFiling:
    p = ParsedFiling(form_type="")

    header_match = HEADER_BLOCK_RE.search(raw)
    if header_match:
        header = header_match.group(1)
        cm = COMPANY_NAME_RE.search(header)
        if cm:
            p.company_name = cm.group(1).strip()
        for m in ITEM_RE.finditer(header):
            p.items.append(m.group(1).strip())

    primary_text = ""
    press_release = ""
    for doc_match in DOCUMENT_RE.finditer(raw):
        doc = doc_match.group(1)
        type_match = TYPE_RE.search(doc)
        if not type_match:
            continue
        doc_type = type_match.group(1).strip().upper()
        text_match = TEXT_RE.search(doc)
        if not text_match:
            continue
        inner = text_match.group(1)
        if "<html" in inner.lower() or "<body" in inner.lower() or "<xbrl" in inner.lower():
            inner = _html_to_text(inner)

        if doc_type.startswith("EX-99"):
            if not press_release:
                press_release = inner
            continue
        if doc_type.startswith("EX-"):
            continue
        if not p.form_type:
            p.form_type = doc_type
        if not primary_text:
            primary_text = inner

    if press_release:
        primary_text += "\n\n[EXHIBIT 99 PRESS RELEASE]\n" + press_release

    primary_text = re.sub(r"&nbsp;", " ", primary_text)
    primary_text = re.sub(r"&amp;", "&", primary_text)
    primary_text = re.sub(r"[ \t]+", " ", primary_text)
    primary_text = re.sub(r"\n{3,}", "\n\n", primary_text).strip()

    if not p.ticker:
        tm = TICKER_RE.search(primary_text[:5000])
        if tm:
            p.ticker = tm.group(1).strip()

    p.body_text = primary_text
    return p
