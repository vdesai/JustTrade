"""CLI: run the 3-layer eval and push results to LangSmith.

Usage:
  python run_eval.py                        # default: 200 sample, 50 with LLM-judge
  python run_eval.py --sample 50            # quick smoke
  python run_eval.py --no-langsmith         # skip LangSmith push
"""

import argparse
import json
from pathlib import Path

from event_bot import config, eval_runner


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x*100:.1f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classified", nargs="+", default=None,
                   help="Paths to classified jsonl files (default: all in data/classified/)")
    p.add_argument("--sample", type=int, default=200,
                   help="How many filings to evaluate")
    p.add_argument("--layer3-subsample", type=int, default=50,
                   help="How many of the sample to also run through LLM-judge (cost)")
    p.add_argument("--no-langsmith", action="store_true",
                   help="Skip pushing to LangSmith dataset")
    args = p.parse_args()

    if args.classified:
        paths = [Path(p) for p in args.classified]
    else:
        paths = sorted(config.CLASSIFIED_DIR.glob("*_oos.jsonl"))
        if not paths:
            paths = sorted(config.CLASSIFIED_DIR.glob("*.jsonl"))
    print(f"Classified inputs ({len(paths)} files):")
    for p in paths:
        print(f"  - {p.name}")

    rows = eval_runner.evaluate(
        paths,
        sample_size=args.sample,
        layer3_subsample=args.layer3_subsample,
    )
    print(f"\n=== Evaluation complete: {len(rows)} rows ===\n")

    agg = eval_runner.aggregate(rows)

    # Top-level summary
    print("LAYER 1 — Event family smoke test (rule-based, weakest)")
    print(f"  evaluable: {agg['layer1']['n_evaluable']:>4d} | "
          f"pass: {agg['layer1']['pass']:>4d} | "
          f"accuracy: {fmt_pct(agg['layer1']['accuracy'])}")
    print()
    print("LAYER 2 — Market-derived sentiment (independent ground truth, strongest)")
    print(f"  evaluable: {agg['layer2']['n_evaluable']:>4d} | "
          f"pass: {agg['layer2']['pass']:>4d} | "
          f"accuracy: {fmt_pct(agg['layer2']['accuracy'])}")
    print()
    if agg["layer2"].get("by_event_type"):
        print("  Layer 2 broken out by event_type (where n>=5):")
        for et, stats in sorted(
            agg["layer2"]["by_event_type"].items(),
            key=lambda x: -x[1]["n"],
        ):
            print(f"    {et:35s} n={stats['n']:>3d}  acc={stats['accuracy']*100:5.1f}%")
        print()

    print("LAYER 3 — Claude Opus as LLM judge (different model, sub-sampled)")
    print(f"  evaluable: {agg['layer3']['n_evaluable']:>4d}")
    print(f"  event_type accuracy: {fmt_pct(agg['layer3']['event_type_accuracy'])}")
    print(f"  sentiment accuracy:  {fmt_pct(agg['layer3']['sentiment_accuracy'])}")
    if agg["layer3"]["avg_rationale_quality"] is not None:
        print(f"  avg rationale quality (1-5): {agg['layer3']['avg_rationale_quality']:.2f}")
    else:
        print(f"  avg rationale quality (1-5): n/a")
    print()

    # Save aggregate
    out = config.DATA_DIR / "eval" / "eval_summary.json"
    out.write_text(json.dumps(agg, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {eval_runner.EVAL_OUTPUT}")

    if not args.no_langsmith:
        try:
            url = eval_runner.push_to_langsmith(rows)
            print(f"\nLangSmith dataset pushed: {url}")
        except Exception as e:
            print(f"\n! LangSmith push failed: {e}")


if __name__ == "__main__":
    main()
