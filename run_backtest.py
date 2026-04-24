import argparse
from pathlib import Path

from event_bot import backtest, config, pipeline, shortability


def main() -> None:
    p = argparse.ArgumentParser(description="Run event-study backtest on classified filings")
    p.add_argument("--classified", nargs="+", default=[str(config.CLASSIFIED_DIR / "classified.jsonl")])
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--min-tradability", type=float, default=0.5)
    p.add_argument("--min-n", type=int, default=5)
    p.add_argument("--horizon", default="ret_5d")
    p.add_argument("--retail-tradable-only", action="store_true",
                   help="Filter out trades where Alpaca can't actually execute (skip unshortables)")
    p.add_argument("--check-shortability", action="store_true",
                   help="Look up shortability for all tickers before running")
    args = p.parse_args()

    events: list = []
    for path in args.classified:
        loaded = pipeline.load_classified(Path(path))
        print(f"  loaded {len(loaded):,} from {path}")
        events.extend(loaded)
    print(f"Total classified events: {len(events):,}")
    if not events:
        print("No events to backtest. Run classify_batch.py first.")
        return

    if args.check_shortability or args.retail_tradable_only:
        unique_tickers = {e.ticker for e in events if e.ticker
                          and e.confidence >= args.min_confidence
                          and e.tradability >= args.min_tradability
                          and e.sentiment != "neutral"}
        shortability.check_tickers(unique_tickers)

    trades = backtest.build_trades(
        events,
        min_confidence=args.min_confidence,
        min_tradability=args.min_tradability,
        retail_tradable_only=args.retail_tradable_only,
    )
    label = " (retail-tradable only)" if args.retail_tradable_only else ""
    print(f"Built {len(trades):,} trades{label}\n")

    in_sample, oos = backtest.in_sample_vs_oos_split(trades, config.OOS_CUTOFF_DATE)
    print(f"IN-SAMPLE (before {config.OOS_CUTOFF_DATE}): {len(in_sample)} trades")
    print(f"OUT-OF-SAMPLE (>= {config.OOS_CUTOFF_DATE}): {len(oos)} trades\n")

    if in_sample:
        print("=" * 80)
        print(f"IN-SAMPLE buckets (horizon={args.horizon}, min n={args.min_n}):")
        print("=" * 80)
        print(backtest.format_report(
            backtest.bucket_stats(in_sample), min_n=args.min_n, horizon=args.horizon
        ))
        print()

    if oos:
        print("=" * 80)
        print(f"OUT-OF-SAMPLE buckets (horizon={args.horizon}, min n={args.min_n}):")
        print("=" * 80)
        print(backtest.format_report(
            backtest.bucket_stats(oos), min_n=args.min_n, horizon=args.horizon
        ))
        print()


if __name__ == "__main__":
    main()
