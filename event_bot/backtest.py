from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev

import pandas as pd
from tqdm import tqdm

from event_bot import config, pipeline, prices, shortability


@dataclass
class Trade:
    ticker: str
    event_type: str
    sentiment: str
    tradability: float
    confidence: float
    date_filed: str
    entry_date: str
    entry_price: float
    returns: dict[str, float]  # "ret_1d", "ret_5d", etc.


@dataclass
class BucketStats:
    event_type: str
    sentiment: str
    n: int
    mean_ret: dict[str, float]  # per horizon
    median_ret: dict[str, float]
    win_rate: dict[str, float]
    sharpe_ann: dict[str, float]


def build_trades(
    events: list[pipeline.ClassifiedEvent],
    min_confidence: float = 0.5,
    min_tradability: float = 0.5,
    retail_tradable_only: bool = False,
) -> list[Trade]:
    shortable_cache = shortability.load_cache() if retail_tradable_only else {}
    trades: list[Trade] = []
    for ev in tqdm(events, desc="Building trades"):
        if not ev.ticker:
            continue
        if ev.confidence < min_confidence or ev.tradability < min_tradability:
            continue
        if retail_tradable_only and not shortability.is_retail_tradable(
            shortable_cache, ev.ticker, ev.sentiment
        ):
            continue
        try:
            r = prices.returns_around_event(ev.ticker, ev.date_filed)
        except Exception:
            continue
        if not r or "entry_price" not in r:
            continue
        ret_keys = {k: v for k, v in r.items() if k.startswith("ret_")}
        if not ret_keys:
            continue

        slip = config.SLIPPAGE_BPS / 10_000
        direction = 1 if ev.sentiment == "bullish" else (-1 if ev.sentiment == "bearish" else 0)
        if direction == 0:
            continue
        adj_returns = {k: (direction * v) - 2 * slip for k, v in ret_keys.items()}

        trades.append(
            Trade(
                ticker=ev.ticker,
                event_type=ev.event_type,
                sentiment=ev.sentiment,
                tradability=ev.tradability,
                confidence=ev.confidence,
                date_filed=ev.date_filed,
                entry_date=r["entry_date"],
                entry_price=r["entry_price"],
                returns=adj_returns,
            )
        )
    return trades


def _sharpe(returns: list[float], trades_per_year: int = 50) -> float:
    if len(returns) < 2:
        return 0.0
    sd = stdev(returns)
    if sd == 0:
        return 0.0
    return (mean(returns) / sd) * (trades_per_year ** 0.5)


def bucket_stats(trades: list[Trade]) -> list[BucketStats]:
    by_bucket: dict[tuple[str, str], list[Trade]] = defaultdict(list)
    for t in trades:
        by_bucket[(t.event_type, t.sentiment)].append(t)

    horizons = sorted({k for t in trades for k in t.returns.keys()})
    out: list[BucketStats] = []
    for (event_type, sentiment), group in by_bucket.items():
        mean_r: dict[str, float] = {}
        median_r: dict[str, float] = {}
        win_r: dict[str, float] = {}
        sharpe_r: dict[str, float] = {}
        for h in horizons:
            rets = [t.returns[h] for t in group if h in t.returns]
            if not rets:
                continue
            mean_r[h] = mean(rets)
            median_r[h] = median(rets)
            win_r[h] = sum(1 for r in rets if r > 0) / len(rets)
            sharpe_r[h] = _sharpe(rets)
        out.append(BucketStats(
            event_type=event_type, sentiment=sentiment, n=len(group),
            mean_ret=mean_r, median_ret=median_r,
            win_rate=win_r, sharpe_ann=sharpe_r,
        ))
    out.sort(key=lambda b: b.n, reverse=True)
    return out


def format_report(stats: list[BucketStats], min_n: int = 5, horizon: str = "ret_5d") -> str:
    lines = []
    header = f"{'EVENT_TYPE':32s} {'SIDE':8s} {'N':>5s} {'MEAN':>8s} {'MEDIAN':>8s} {'WIN%':>6s} {'SHARPE':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    for b in stats:
        if b.n < min_n or horizon not in b.mean_ret:
            continue
        lines.append(
            f"{b.event_type:32s} {b.sentiment:8s} {b.n:>5d} "
            f"{b.mean_ret[horizon]*100:>7.2f}% {b.median_ret[horizon]*100:>7.2f}% "
            f"{b.win_rate[horizon]*100:>5.1f}% {b.sharpe_ann[horizon]:>7.2f}"
        )
    return "\n".join(lines)


def in_sample_vs_oos_split(
    trades: list[Trade], oos_cutoff: str
) -> tuple[list[Trade], list[Trade]]:
    in_sample = [t for t in trades if t.date_filed < oos_cutoff]
    oos = [t for t in trades if t.date_filed >= oos_cutoff]
    return in_sample, oos
