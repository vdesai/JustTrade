import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

import config

ROOT = Path(__file__).parent
STATE_FILE = ROOT / "state.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    with open(LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log", "a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"peak_equity": None, "killed": False, "entries": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def get_bars(data_client: StockHistoricalDataClient, symbol: str) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=400)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = data_client.get_stock_bars(req).df
    if bars.empty:
        return bars
    if "symbol" in bars.index.names:
        bars = bars.xs(symbol, level="symbol")
    return bars


def compute_signals(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["rsi"] = rsi(df["close"], config.RSI_PERIOD)
    df["ma_trend"] = df["close"].rolling(config.TREND_MA).mean()
    df["ma_exit"] = df["close"].rolling(config.EXIT_MA).mean()
    df["atr"] = atr(df, config.ATR_PERIOD)
    last = df.iloc[-1]
    return {
        "close": float(last["close"]),
        "rsi": float(last["rsi"]),
        "ma_trend": float(last["ma_trend"]),
        "ma_exit": float(last["ma_exit"]),
        "atr": float(last["atr"]),
    }


def position_size(equity: float, price: float, atr_val: float) -> int:
    stop_distance = config.ATR_STOP_MULT * atr_val
    if stop_distance <= 0:
        return 0
    risk_dollars = equity * config.RISK_PER_TRADE
    shares = int(risk_dollars / stop_distance)
    max_notional = equity * 0.33
    max_shares_by_notional = int(max_notional / price)
    return max(0, min(shares, max_shares_by_notional))


def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.getenv("ALPACA_KEY")
    secret = os.getenv("ALPACA_SECRET")
    live = os.getenv("LIVE", "false").lower() == "true"

    if not key or not secret:
        log("ERROR: ALPACA_KEY / ALPACA_SECRET missing from .env")
        sys.exit(1)

    mode = "LIVE" if live else "PAPER"
    log(f"=== Bot run starting in {mode} mode ===")

    trading = TradingClient(key, secret, paper=not live)
    data = StockHistoricalDataClient(key, secret)

    account = trading.get_account()
    equity = float(account.equity)
    log(f"Account equity: ${equity:.2f} | buying power: ${float(account.buying_power):.2f}")

    state = load_state()
    if state["peak_equity"] is None:
        state["peak_equity"] = equity
    state["peak_equity"] = max(state["peak_equity"], equity)
    drawdown = 1 - (equity / state["peak_equity"])

    if state["killed"]:
        log(f"KILL SWITCH ACTIVE. Peak=${state['peak_equity']:.2f}, now=${equity:.2f}. Exiting.")
        return

    if drawdown >= config.KILL_SWITCH_DRAWDOWN:
        state["killed"] = True
        save_state(state)
        log(f"KILL SWITCH TRIGGERED: drawdown {drawdown:.1%} >= {config.KILL_SWITCH_DRAWDOWN:.0%}")
        log("Closing all open positions.")
        trading.close_all_positions(cancel_orders=True)
        return

    positions = {p.symbol: p for p in trading.get_all_positions()}
    open_count = len(positions)
    log(f"Open positions: {open_count} ({list(positions.keys())})")

    for symbol in config.UNIVERSE:
        try:
            bars = get_bars(data, symbol)
            if len(bars) < config.TREND_MA + 5:
                log(f"{symbol}: insufficient history ({len(bars)} bars), skipping")
                continue
            sig = compute_signals(bars)
            log(
                f"{symbol}: close={sig['close']:.2f} rsi={sig['rsi']:.1f} "
                f"ma200={sig['ma_trend']:.2f} ma5={sig['ma_exit']:.2f} atr={sig['atr']:.2f}"
            )

            if symbol in positions:
                pos = positions[symbol]
                entry_price = state["entries"].get(symbol, float(pos.avg_entry_price))
                stop_price = entry_price - config.ATR_STOP_MULT * sig["atr"]
                should_exit = sig["close"] > sig["ma_exit"]
                stop_hit = sig["close"] < stop_price
                if should_exit or stop_hit:
                    reason = "exit (above 5MA)" if should_exit else f"stop (< {stop_price:.2f})"
                    log(f"{symbol}: SELL {pos.qty} shares — {reason}")
                    trading.submit_order(MarketOrderRequest(
                        symbol=symbol,
                        qty=float(pos.qty),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    ))
                    state["entries"].pop(symbol, None)
            else:
                if open_count >= config.MAX_OPEN_POSITIONS:
                    continue
                uptrend = sig["close"] > sig["ma_trend"]
                oversold = sig["rsi"] < config.RSI_OVERSOLD
                if uptrend and oversold:
                    qty = position_size(equity, sig["close"], sig["atr"])
                    if qty < 1:
                        log(f"{symbol}: sized to 0 shares (atr too high for risk budget)")
                        continue
                    log(f"{symbol}: BUY {qty} shares @ ~${sig['close']:.2f} (risk 1% of ${equity:.0f})")
                    trading.submit_order(MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    ))
                    state["entries"][symbol] = sig["close"]
                    open_count += 1
        except Exception as e:
            log(f"{symbol}: ERROR — {e}")

    save_state(state)
    log("=== Bot run complete ===\n")


if __name__ == "__main__":
    main()
