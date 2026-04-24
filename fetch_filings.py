import argparse
import sys

from tqdm import tqdm

from event_bot import edgar


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch SEC filings index + (optionally) bodies")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--forms", nargs="+", default=["8-K"])
    p.add_argument("--download-bodies", action="store_true",
                   help="Download full filing text (slow, rate-limited)")
    p.add_argument("--limit", type=int, default=None, help="Cap number of filings (for testing)")
    args = p.parse_args()

    print(f"Fetching {args.forms} filings from {args.start} to {args.end}...")
    filings = edgar.fetch_date_range(args.start, args.end, forms=args.forms)
    print(f"Found {len(filings):,} filings in index")

    if args.limit:
        filings = filings[: args.limit]
        print(f"Limiting to first {len(filings)} for this run")

    if args.download_bodies:
        for f in tqdm(filings, desc="Downloading bodies"):
            try:
                edgar.fetch_filing_text(f)
            except Exception as e:
                print(f"  ! {f.accession}: {e}", file=sys.stderr)

    print(f"\nSample of first 5 filings:")
    for f in filings[:5]:
        print(f"  {f.date_filed}  {f.form_type:6s}  {f.cik:>10s}  {f.company_name[:60]}")


if __name__ == "__main__":
    main()
