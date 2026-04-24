import argparse
from pathlib import Path

from event_bot import config, pipeline, shortability, walk_forward


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward validation")
    p.add_argument("--classified", nargs="+", required=True)
    p.add_argument("--train-months", type=int, default=6)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--min-n-train", type=int, default=15)
    p.add_argument("--min-sharpe", type=float, default=0.5)
    p.add_argument("--min-mean-return", type=float, default=0.005)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--min-tradability", type=float, default=0.5)
    p.add_argument("--horizon", default="ret_5d")
    p.add_argument("--retail-tradable-only", action="store_true")
    args = p.parse_args()

    events = []
    for path in args.classified:
        events.extend(pipeline.load_classified(Path(path)))
    print(f"Loaded {len(events):,} classified events")

    if args.retail_tradable_only:
        tickers = {e.ticker for e in events if e.ticker
                   and e.confidence >= args.min_confidence
                   and e.tradability >= args.min_tradability
                   and e.sentiment != "neutral"}
        shortability.check_tickers(tickers)

    results = walk_forward.walk_forward(
        events,
        train_months=args.train_months,
        test_months=args.test_months,
        min_confidence=args.min_confidence,
        min_tradability=args.min_tradability,
        min_n_train=args.min_n_train,
        min_sharpe=args.min_sharpe,
        min_mean_return=args.min_mean_return,
        horizon=args.horizon,
        retail_tradable_only=args.retail_tradable_only,
    )
    print()
    print(walk_forward.summarize(results))


if __name__ == "__main__":
    main()
