"""Check which tickers Alpaca will let us short (paper account)."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from alpaca.trading.client import TradingClient

from event_bot import config

load_dotenv(config.ROOT / ".env")

_CACHE = config.DATA_DIR / "shortable_map.json"


def _client() -> TradingClient:
    return TradingClient(os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET"), paper=True)


def load_cache() -> dict[str, dict]:
    if _CACHE.exists():
        return json.loads(_CACHE.read_text())
    return {}


def save_cache(cache: dict[str, dict]) -> None:
    _CACHE.write_text(json.dumps(cache, indent=2))


def check_tickers(tickers: set[str]) -> dict[str, dict]:
    cache = load_cache()
    todo = [t for t in sorted(tickers) if t not in cache]
    if not todo:
        print(f"All {len(tickers)} tickers already cached")
        return cache

    print(f"Looking up {len(todo)} tickers on Alpaca (cached: {len(cache)})")
    trading = _client()
    for i, symbol in enumerate(tqdm(todo, desc="Checking shortability")):
        try:
            asset = trading.get_asset(symbol)
            cache[symbol] = {
                "tradable": bool(getattr(asset, "tradable", False)),
                "shortable": bool(getattr(asset, "shortable", False)),
                "easy_to_borrow": bool(getattr(asset, "easy_to_borrow", False)),
                "fractionable": bool(getattr(asset, "fractionable", False)),
                "status": str(getattr(asset, "status", "unknown")),
            }
        except Exception as e:
            cache[symbol] = {"tradable": False, "shortable": False, "error": str(e)[:100]}
        if (i + 1) % 50 == 0:
            save_cache(cache)
        time.sleep(0.05)
    save_cache(cache)
    return cache


def is_retail_tradable(cache: dict[str, dict], ticker: str, direction: str) -> bool:
    info = cache.get(ticker)
    if not info or not info.get("tradable"):
        return False
    if direction == "bearish":
        return bool(info.get("shortable"))
    return True
