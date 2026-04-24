import argparse

from event_bot import config, edgar, pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Classify SEC 8-K filings with Claude")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--output", default=str(config.CLASSIFIED_DIR / "classified.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--forms", nargs="+", default=["8-K"])
    args = p.parse_args()

    print(f"Fetching index {args.start} → {args.end}")
    filings = edgar.fetch_date_range(args.start, args.end, forms=args.forms)
    print(f"{len(filings):,} filings in range")
    if args.limit:
        filings = filings[: args.limit]
        print(f"Limiting to {len(filings)}")

    from pathlib import Path

    results = pipeline.classify_batch(filings, Path(args.output))
    print(f"\nClassified {len(results)} new filings. Total written to {args.output}")


if __name__ == "__main__":
    main()
