import json
from pathlib import Path

import requests

from event_bot import config
from event_bot.edgar import _get

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_TICKER_CACHE = config.DATA_DIR / "cik_to_ticker.json"


def load_cik_to_ticker() -> dict[str, str]:
    if _TICKER_CACHE.exists():
        return json.loads(_TICKER_CACHE.read_text())
    r = _get(TICKER_MAP_URL)
    data = r.json()
    mapping: dict[str, str] = {}
    for row in data.values():
        cik = str(int(row["cik_str"]))
        mapping[cik] = row["ticker"]
    _TICKER_CACHE.write_text(json.dumps(mapping))
    return mapping
