import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv

from event_bot import config

load_dotenv(config.ROOT / ".env")

SEC_BASE = "https://www.sec.gov"
FULL_INDEX_URL = SEC_BASE + "/Archives/edgar/full-index/{year}/QTR{q}/form.idx"


def _user_agent() -> str:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise RuntimeError(
            "SEC_USER_AGENT missing in .env — SEC requires 'Name email@example.com'"
        )
    return ua


class _RateLimiter:
    def __init__(self, per_sec: int) -> None:
        self.interval = 1.0 / per_sec
        self.last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last = time.monotonic()


_limiter = _RateLimiter(config.SEC_RATE_LIMIT_PER_SEC)


def _get(url: str) -> requests.Response:
    _limiter.wait()
    r = requests.get(url, headers={"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    return r


@dataclass
class Filing:
    cik: str
    company_name: str
    form_type: str
    date_filed: str
    accession: str
    filename: str

    @property
    def primary_doc_url(self) -> str:
        return f"{SEC_BASE}/Archives/{self.filename}"

    @property
    def local_path(self) -> Path:
        safe = self.accession.replace("/", "_")
        return config.FILINGS_DIR / f"{self.date_filed}_{self.cik}_{safe}.txt"


_IDX_RE = re.compile(
    r"^(?P<form>\S+(?:\s\S+)*?)\s{2,}(?P<company>.+?)\s{2,}(?P<cik>\d+)\s{2,}(?P<date>\d{4}-\d{2}-\d{2})\s{2,}(?P<filename>\S+)\s*$"
)


def fetch_quarter_index(year: int, quarter: int, forms: Iterable[str] = None) -> list[Filing]:
    forms = set(forms or config.FORM_TYPES)
    cache = config.INDEXES_DIR / f"form_{year}_Q{quarter}.idx"

    # Current quarter's data changes daily — always refetch if cache >6h old.
    # Past quarters are immutable — use cache forever.
    now = datetime.now()
    current_year, current_q = now.year, (now.month - 1) // 3 + 1
    is_current_quarter = (year, quarter) == (current_year, current_q)
    use_cache = cache.exists()
    if is_current_quarter and use_cache:
        age_hours = (now.timestamp() - cache.stat().st_mtime) / 3600
        if age_hours > 6:
            use_cache = False

    if use_cache:
        text = cache.read_text()
    else:
        url = FULL_INDEX_URL.format(year=year, q=quarter)
        text = _get(url).text
        cache.write_text(text)

    filings: list[Filing] = []
    for line in text.splitlines():
        m = _IDX_RE.match(line)
        if not m:
            continue
        form = m.group("form").strip()
        if form not in forms:
            continue
        fn = m.group("filename").strip()
        accession = fn.split("/")[-1].replace(".txt", "")
        filings.append(
            Filing(
                cik=m.group("cik").strip().lstrip("0") or "0",
                company_name=m.group("company").strip(),
                form_type=form,
                date_filed=m.group("date").strip(),
                accession=accession,
                filename=fn,
            )
        )
    return filings


def fetch_filing_text(filing: Filing, use_cache: bool = True) -> str:
    if use_cache and filing.local_path.exists():
        return filing.local_path.read_text(errors="replace")
    text = _get(filing.primary_doc_url).text
    filing.local_path.write_text(text)
    return text


def fetch_date_range(start: str, end: str, forms: Iterable[str] = None) -> list[Filing]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    quarters: set[tuple[int, int]] = set()
    cur = start_dt
    while cur <= end_dt:
        quarters.add((cur.year, (cur.month - 1) // 3 + 1))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    out: list[Filing] = []
    for year, q in sorted(quarters):
        for f in fetch_quarter_index(year, q, forms=forms):
            if start <= f.date_filed <= end:
                out.append(f)
    return out
