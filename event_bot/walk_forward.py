"""Walk-forward validation — the proper way to validate a trading strategy.

Instead of one big in-sample/out-of-sample split (which lets us tune
thresholds knowing the in-sample results), we:
  1. Define a rolling window: train on N months, test on next M months
  2. For each window: pick "promising" buckets using only training data
  3. Apply those picks to the test window, measure performance
  4. Slide forward by M months, repeat
  5. Aggregate test-window performance across all windows

A strategy that shows edge *consistently* across many rolling test windows
is far more trustworthy than one that shows edge only on a single split.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, stdev
from typing import Iterable

from event_bot import backtest, pipeline


@dataclass
class WindowResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_buckets: list[tuple[str, str]]  # (event_type, our_action)
    train_stats: dict[tuple[str, str], dict[str, float]]
    test_trades: list[backtest.Trade]
    test_mean_return: float
    test_sharpe: float
    test_win_rate: float
    test_n: int


def _months_between(start: str, end: str) -> list[str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out: list[str] = []
    cur = s.replace(day=1)
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return out


def _month_end(month_start: str) -> str:
    dt = datetime.strptime(month_start, "%Y-%m-%d")
    if dt.month == 12:
        nxt = dt.replace(year=dt.year + 1, month=1)
    else:
        nxt = dt.replace(month=dt.month + 1)
    from datetime import timedelta
    return (nxt - timedelta(days=1)).strftime("%Y-%m-%d")


def select_buckets_from_train(
    train_trades: list[backtest.Trade],
    min_n: int,
    min_sharpe: float,
    min_mean_return: float,
    horizon: str = "ret_5d",
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], dict[str, float]]]:
    """Given training trades, pick buckets (event_type, our_action) worth trading.

    Returns both the picks AND the full stats (for reporting/debugging).
    """
    # Group by (event_type, sentiment_we_bet)
    # We consider BOTH directions (agree with classifier OR contrarian) for each bucket.
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for t in train_trades:
        if horizon not in t.returns:
            continue
        r = t.returns[horizon]
        # Original direction (we bet the classifier's call)
        grouped[(t.event_type, t.sentiment, "agree")].append(r)
        # Contrarian (we bet opposite of classifier)
        grouped[(t.event_type, t.sentiment, "contra")].append(-r)

    stats: dict[tuple[str, str], dict[str, float]] = {}
    picks: list[tuple[str, str]] = []

    # For each (event_type, classifier_sentiment) choose the better of agree/contra
    by_bucket: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(dict)
    for (et, sent, mode), rets in grouped.items():
        by_bucket[(et, sent)][mode] = rets

    for (et, sent), modes in by_bucket.items():
        best_mode, best_mean, best_sharpe, best_n = None, 0.0, 0.0, 0
        for mode, rets in modes.items():
            n = len(rets)
            if n < min_n:
                continue
            m = mean(rets)
            sd = stdev(rets) if n > 1 else 1e-9
            sharpe = (m / sd) * (50 ** 0.5) if sd > 0 else 0.0
            if m > best_mean:
                best_mode, best_mean, best_sharpe, best_n = mode, m, sharpe, n
        if best_mode is None:
            continue
        our_action = "long" if (sent == "bullish" and best_mode == "agree") or \
                               (sent == "bearish" and best_mode == "contra") else "short"
        stats[(et, sent)] = {
            "mode": best_mode,
            "our_action": our_action,
            "n": best_n,
            "mean": best_mean,
            "sharpe": best_sharpe,
        }
        if best_mean >= min_mean_return and best_sharpe >= min_sharpe:
            picks.append((et, sent))
    return picks, stats


def apply_picks_to_test(
    test_trades_raw: list[backtest.Trade],
    picks_with_actions: dict[tuple[str, str], str],
    horizon: str,
) -> list[backtest.Trade]:
    """Filter test trades to only those matching selected buckets, with correct direction."""
    out: list[backtest.Trade] = []
    for t in test_trades_raw:
        action = picks_with_actions.get((t.event_type, t.sentiment))
        if action is None:
            continue
        if horizon not in t.returns:
            continue
        # Test trade returns are direction-adjusted by sentiment. If we agreed
        # with classifier → return is as-is. If contrarian → flip sign.
        if (action == "long" and t.sentiment == "bullish") or \
           (action == "short" and t.sentiment == "bearish"):
            adjusted = dict(t.returns)  # agree
        else:
            adjusted = {k: -v for k, v in t.returns.items()}  # contrarian flip
        new_t = backtest.Trade(
            ticker=t.ticker, event_type=t.event_type, sentiment=t.sentiment,
            tradability=t.tradability, confidence=t.confidence,
            date_filed=t.date_filed, entry_date=t.entry_date,
            entry_price=t.entry_price, returns=adjusted,
        )
        out.append(new_t)
    return out


def walk_forward(
    events: list[pipeline.ClassifiedEvent],
    train_months: int = 6,
    test_months: int = 1,
    min_confidence: float = 0.5,
    min_tradability: float = 0.5,
    min_n_train: int = 15,
    min_sharpe: float = 0.5,
    min_mean_return: float = 0.005,
    horizon: str = "ret_5d",
    retail_tradable_only: bool = True,
    data_start: str | None = None,
    data_end: str | None = None,
) -> list[WindowResult]:
    """Roll through time, train/test at each step."""
    events_sorted = sorted(events, key=lambda e: e.date_filed)
    if not events_sorted:
        return []
    data_start = data_start or events_sorted[0].date_filed[:7] + "-01"
    data_end = data_end or events_sorted[-1].date_filed
    month_starts = _months_between(data_start, data_end)
    if len(month_starts) < train_months + test_months:
        return []

    # Build trades once
    all_trades = backtest.build_trades(
        events_sorted, min_confidence=min_confidence,
        min_tradability=min_tradability,
        retail_tradable_only=retail_tradable_only,
    )
    trades_by_month: dict[str, list[backtest.Trade]] = defaultdict(list)
    for t in all_trades:
        trades_by_month[t.date_filed[:7]].append(t)

    results: list[WindowResult] = []
    for i in range(len(month_starts) - train_months - test_months + 1):
        train_months_list = month_starts[i : i + train_months]
        test_months_list = month_starts[i + train_months : i + train_months + test_months]

        train_trades = []
        for m in train_months_list:
            train_trades.extend(trades_by_month.get(m[:7], []))
        test_trades = []
        for m in test_months_list:
            test_trades.extend(trades_by_month.get(m[:7], []))

        if not train_trades or not test_trades:
            continue

        picks, train_stats = select_buckets_from_train(
            train_trades,
            min_n=min_n_train,
            min_sharpe=min_sharpe,
            min_mean_return=min_mean_return,
            horizon=horizon,
        )
        picks_actions: dict[tuple[str, str], str] = {
            k: train_stats[k]["our_action"] for k in picks
        }
        applied = apply_picks_to_test(test_trades, picks_actions, horizon)

        if applied:
            rets = [t.returns[horizon] for t in applied if horizon in t.returns]
            m = mean(rets) if rets else 0.0
            sd = stdev(rets) if len(rets) > 1 else 1e-9
            sh = (m / sd) * (50 ** 0.5) if sd > 0 else 0.0
            wr = sum(1 for r in rets if r > 0) / len(rets) if rets else 0.0
        else:
            m = sh = wr = 0.0

        results.append(WindowResult(
            train_start=train_months_list[0],
            train_end=_month_end(train_months_list[-1]),
            test_start=test_months_list[0],
            test_end=_month_end(test_months_list[-1]),
            selected_buckets=picks,
            train_stats=train_stats,
            test_trades=applied,
            test_mean_return=m,
            test_sharpe=sh,
            test_win_rate=wr,
            test_n=len(applied),
        ))
    return results


def summarize(results: list[WindowResult]) -> str:
    if not results:
        return "No walk-forward windows produced."
    lines = []
    lines.append(f"{'TRAIN':23s} {'TEST':23s} {'PICKS':>6s} {'N':>5s} {'MEAN':>8s} {'WIN%':>6s} {'SHARPE':>7s}")
    lines.append("-" * 90)
    for r in results:
        lines.append(
            f"{r.train_start}→{r.train_end[:10]:12s} "
            f"{r.test_start}→{r.test_end[:10]:12s} "
            f"{len(r.selected_buckets):>6d} {r.test_n:>5d} "
            f"{r.test_mean_return*100:>7.2f}% {r.test_win_rate*100:>5.1f}% "
            f"{r.test_sharpe:>6.2f}"
        )
    total_n = sum(r.test_n for r in results)
    all_rets = [t.returns.get("ret_5d", 0) for r in results for t in r.test_trades]
    if all_rets:
        from statistics import mean as _m, stdev as _s
        mr = _m(all_rets)
        sd = _s(all_rets) if len(all_rets) > 1 else 1e-9
        sh = (mr / sd) * (50 ** 0.5) if sd > 0 else 0.0
        wr = sum(1 for r in all_rets if r > 0) / len(all_rets)
        lines.append("-" * 90)
        lines.append(
            f"{'AGGREGATE':45s}   {'':>6s} {total_n:>5d} {mr*100:>7.2f}% {wr*100:>5.1f}% {sh:>6.2f}"
        )
    return "\n".join(lines)
