"""Live paper-trading executor for event-driven signals.

Designed to be run once per day (morning, after market opens). It:
1. Fetches 8-K filings from the past N days that haven't been processed
2. Classifies them with the LLM
3. For each signal passing the strategy filter, submits a market order to Alpaca paper
4. Exits positions at their configured holding-period end
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from event_bot import config, edgar, pipeline

TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

load_dotenv(config.ROOT / ".env")

STATE_FILE = config.DATA_DIR / "paper_trader_state.json"
LIVE_LOG = config.DATA_DIR / "paper_trader.log"


@dataclass
class OpenPosition:
    symbol: str
    qty: float
    entry_date: str
    exit_date: str
    event_type: str
    filing_accession: str
    sentiment: str


def _log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    with open(LIVE_LOG, "a") as f:
        f.write(line + "\n")


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"seen_accessions": [], "open_positions": []}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _make_client() -> TradingClient:
    key = os.getenv("ALPACA_KEY")
    secret = os.getenv("ALPACA_SECRET")
    live = os.getenv("LIVE", "false").lower() == "true"
    if not key or not secret:
        raise RuntimeError("Alpaca keys missing in .env")
    return TradingClient(key, secret, paper=not live)


def _make_data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET"))


def _asset_is_shortable(trading: TradingClient, symbol: str) -> tuple[bool, str]:
    try:
        asset = trading.get_asset(symbol)
    except Exception as e:
        return False, f"lookup_error: {e}"
    if not getattr(asset, "tradable", False):
        return False, "not tradable"
    if not getattr(asset, "shortable", False):
        return False, "not shortable (HTB or restricted)"
    return True, "ok"


def _latest_price(data: StockHistoricalDataClient, symbol: str) -> float | None:
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        resp = data.get_stock_latest_trade(req)
        return float(resp[symbol].price)
    except Exception:
        return None


def _next_business_day(d: str) -> str:
    dt = datetime.strptime(d, "%Y-%m-%d")
    # crude: skip Sat/Sun; doesn't handle US market holidays
    while True:
        dt += timedelta(days=1)
        if dt.weekday() < 5:
            return dt.strftime("%Y-%m-%d")


def _holding_exit(entry_date: str, holding_days: int) -> str:
    exit_date = entry_date
    for _ in range(holding_days):
        exit_date = _next_business_day(exit_date)
    return exit_date


def process_exits(trading: TradingClient, state: dict, today: str) -> None:
    remaining = []
    for raw_pos in state["open_positions"]:
        pos = OpenPosition(**raw_pos)
        if today >= pos.exit_date:
            _log(f"EXIT {pos.symbol} {pos.qty} sh (held since {pos.entry_date}, reason=holding expired)")
            try:
                trading.submit_order(MarketOrderRequest(
                    symbol=pos.symbol,
                    qty=pos.qty,
                    side=OrderSide.SELL if pos.sentiment in ("long", "bullish") else OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                ))
            except Exception as e:
                _log(f"  ! exit order failed: {e}")
        else:
            remaining.append(raw_pos)
    state["open_positions"] = remaining


def process_new_signals(
    trading: TradingClient,
    state: dict,
    lookback_days: int,
    min_confidence: float,
    min_tradability: float,
    risk_per_trade: float,
    holding_days: int,
    strategy_map: dict[tuple[str, str], str] | None = None,
) -> None:
    account = trading.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    _log(f"Account equity=${equity:.2f} buying_power=${buying_power:.2f}")

    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    _log(f"Scanning 8-Ks from {start} to {today}")

    filings = edgar.fetch_date_range(start, today, forms=["8-K"])
    seen = set(state["seen_accessions"])
    new_filings = [f for f in filings if f.accession not in seen]
    _log(f"{len(new_filings)} new filings to classify ({len(filings) - len(new_filings)} already seen)")

    from event_bot import tickers as tickers_mod
    cik_to_ticker = tickers_mod.load_cik_to_ticker()
    data_client = _make_data_client()

    for filing in new_filings:
        if filing.cik not in cik_to_ticker:
            seen.add(filing.accession)
            continue
        try:
            ev = pipeline.classify_filing(filing, cik_to_ticker)
        except Exception as e:
            _log(f"  ! classify failed for {filing.accession}: {e}")
            continue
        seen.add(filing.accession)
        if ev is None:
            continue

        passes = (
            ev.confidence >= min_confidence
            and ev.tradability >= min_tradability
            and ev.sentiment in ("bullish", "bearish")
        )
        our_action: str | None = None
        if strategy_map is not None:
            our_action = strategy_map.get((ev.event_type, ev.sentiment))
            passes = passes and our_action in ("long", "short")
        if not passes:
            continue

        if not TICKER_RE.match(ev.ticker):
            _log(f"  skip {ev.ticker}: non-standard ticker (warrant/unit/pref)")
            continue

        ok, reason = _asset_is_shortable(trading, ev.ticker)
        if our_action == "short" and not ok:
            _log(f"  skip {ev.ticker}: {reason}")
            continue

        price = _latest_price(data_client, ev.ticker)
        if price is None or price < 1.0:
            _log(f"  skip {ev.ticker}: no recent price or < $1 (price={price})")
            continue

        risk_dollars = equity * risk_per_trade
        notional_cap = equity * 0.05
        target_notional = min(risk_dollars * 5, notional_cap)
        qty = int(target_notional / price)
        if qty < 1:
            _log(f"  skip {ev.ticker}: qty rounds to 0 (price={price:.2f}, target=${target_notional:.0f})")
            continue

        direction = OrderSide.BUY if our_action == "long" else OrderSide.SELL
        _log(
            f"SIGNAL {ev.ticker} {ev.event_type}/{ev.sentiment} → {our_action.upper()} "
            f"conf={ev.confidence:.2f} trade={ev.tradability:.2f} "
            f"→ {qty} sh @ ~${price:.2f} (${qty*price:.0f})"
        )
        try:
            trading.submit_order(MarketOrderRequest(
                symbol=ev.ticker,
                qty=qty,
                side=direction,
                time_in_force=TimeInForce.DAY,
            ))
            entry_date = today
            exit_date = _holding_exit(entry_date, holding_days)
            state["open_positions"].append(asdict(OpenPosition(
                symbol=ev.ticker,
                qty=float(qty),
                entry_date=entry_date,
                exit_date=exit_date,
                event_type=ev.event_type,
                filing_accession=filing.accession,
                sentiment=our_action,  # "long" or "short" — how we entered
            )))
        except Exception as e:
            _log(f"  ! order failed: {e}")

    state["seen_accessions"] = sorted(seen)


def run_once(
    lookback_days: int = 1,
    min_confidence: float = 0.7,
    min_tradability: float = 0.7,
    risk_per_trade: float = 0.01,
    holding_days: int = 5,
    strategy_map: dict[tuple[str, str], str] | None = None,
) -> None:
    state = _load_state()
    trading = _make_client()
    today = datetime.now().strftime("%Y-%m-%d")
    _log(f"=== paper_trader run {today} ===")
    process_exits(trading, state, today)
    process_new_signals(
        trading, state,
        lookback_days=lookback_days,
        min_confidence=min_confidence,
        min_tradability=min_tradability,
        risk_per_trade=risk_per_trade,
        holding_days=holding_days,
        strategy_map=strategy_map,
    )
    _save_state(state)
    _log("=== run complete ===\n")


# Validated via Q1 2026 OOS backtest, market-adjusted.
# Key: (event_type_from_classifier, sentiment_from_classifier) -> our_action
# "short" means we bet against the classifier's bullish calls (contrarian),
# "long" means we agree with the classifier.
STRATEGY_MAP = {
    ("exec_appointment", "bullish"): "short",
    ("buyback_announced", "bullish"): "short",
    ("fda_approval", "bullish"): "short",
    ("restructuring_layoffs", "bearish"): "short",
}


if __name__ == "__main__":
    run_once(strategy_map=STRATEGY_MAP)
