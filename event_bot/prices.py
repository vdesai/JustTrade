from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from event_bot import config


def _cache_path(ticker: str) -> Path:
    return config.PRICES_DIR / f"{ticker.upper()}.pkl"


def get_prices(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    path = _cache_path(ticker)
    if use_cache and path.exists():
        df = pd.read_pickle(path)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if df.index.min() <= s and df.index.max() >= e:
            return df.loc[s:e]

    start_pad = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end_pad = (pd.Timestamp(end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    df = yf.download(
        ticker,
        start=start_pad,
        end=end_pad,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if use_cache:
        df.to_pickle(path)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return df.loc[s:e]


def returns_around_event(
    ticker: str, event_date: str, holding_days: list[int] | None = None
) -> dict[str, float]:
    holding_days = holding_days or config.HOLDING_PERIODS_DAYS
    max_h = max(holding_days)
    start = (pd.Timestamp(event_date) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(event_date) + pd.Timedelta(days=max_h + 10)).strftime("%Y-%m-%d")
    px = get_prices(ticker, start, end)
    if px.empty or "Close" not in px.columns:
        return {}

    event_ts = pd.Timestamp(event_date)
    future = px[px.index >= event_ts]
    if future.empty:
        return {}

    entry_idx = future.index[0]
    if len(future) < 2:
        return {}
    entry_price = float(future.iloc[1]["Open"]) if "Open" in future.columns else float(future.iloc[1]["Close"])

    out: dict[str, float] = {"entry_price": entry_price, "entry_date": str(future.index[1].date())}
    for h in holding_days:
        if len(future) <= h + 1:
            continue
        exit_price = float(future.iloc[h + 1]["Close"])
        out[f"ret_{h}d"] = (exit_price / entry_price) - 1
    return out
