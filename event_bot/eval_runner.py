"""Run the 3-layer eval over a sample of classified filings, aggregate metrics,
push to LangSmith as a Dataset + Experiment.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from langsmith import Client

from event_bot import (anonymize, config, edgar, eval_layers, parser,
                       pipeline, tickers)

load_dotenv(config.ROOT / ".env")

EVAL_OUTPUT = config.DATA_DIR / "eval" / "eval_results.jsonl"
EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "justtrade-classifier-eval-q1-2026"


@dataclass
class EvalRow:
    accession: str
    ticker: str
    date_filed: str
    items: list[str]
    classifier_event_type: str
    classifier_sentiment: str
    classifier_confidence: float
    classifier_rationale: str
    layer1_passed: bool
    layer1_reason: str
    layer2_passed: bool
    layer2_market_sentiment: str
    layer2_return_5d: float | None
    layer2_reason: str
    # Layer 3 only run on subset (cost)
    layer3_run: bool = False
    layer3_event_type_correct: bool = False
    layer3_sentiment_correct: bool = False
    layer3_rationale_quality: int = 0
    layer3_reason: str = ""


def _load_filing_text(accession: str) -> tuple[str, list[str], str]:
    """Find the cached filing, parse it, return (anonymized_body, items, company_name)."""
    fpath = next(config.FILINGS_DIR.glob(f"*_{accession}.txt"), None)
    if fpath is None:
        return "", [], ""
    raw = fpath.read_text(errors="replace")
    parsed = parser.parse_filing(raw)
    return parsed.body_text, parsed.items, parsed.company_name


def evaluate(
    classified_paths: list[Path],
    sample_size: int = 200,
    layer3_subsample: int = 50,
    seed: int = 42,
) -> list[EvalRow]:
    """Run the 3-layer eval. Layer 3 (LLM judge) only on `layer3_subsample` rows for cost."""
    cik_to_ticker = tickers.load_cik_to_ticker()

    events: list[pipeline.ClassifiedEvent] = []
    for p in classified_paths:
        events.extend(pipeline.load_classified(p))

    # Filter to events whose filings we have locally cached + have a ticker
    cached = {p.stem.rsplit("_", 1)[-1] for p in config.FILINGS_DIR.glob("*.txt")}
    have = [e for e in events if e.ticker and e.accession in cached]
    print(f"  candidates: {len(events):,} classified → {len(have):,} with cached body + ticker")
    if not have:
        return []

    random.Random(seed).shuffle(have)
    sample = have[:sample_size]
    layer3_set = set(e.accession for e in random.Random(seed + 1).sample(sample, min(layer3_subsample, len(sample))))

    rows: list[EvalRow] = []
    out_f = open(EVAL_OUTPUT, "w")
    try:
        for ev in tqdm(sample, desc="Evaluating"):
            body_raw, items, company = _load_filing_text(ev.accession)
            if not body_raw:
                continue
            anon = anonymize.anonymize(body_raw, company_name=company, ticker=ev.ticker)

            l1 = eval_layers.grade_layer1_smoke_test(ev.event_type, items)
            l2 = eval_layers.grade_layer2_market_truth(ev.ticker, ev.date_filed, ev.sentiment)

            row = EvalRow(
                accession=ev.accession, ticker=ev.ticker, date_filed=ev.date_filed,
                items=items,
                classifier_event_type=ev.event_type,
                classifier_sentiment=ev.sentiment,
                classifier_confidence=ev.confidence,
                classifier_rationale=ev.rationale,
                layer1_passed=l1.passed, layer1_reason=l1.reason,
                layer2_passed=l2.passed, layer2_market_sentiment=l2.market_sentiment,
                layer2_return_5d=l2.return_5d, layer2_reason=l2.reason,
            )

            if ev.accession in layer3_set:
                l3 = eval_layers.grade_layer3_llm_judge(
                    anon, items, ev.event_type, ev.sentiment, ev.rationale,
                    accession=ev.accession,
                )
                row.layer3_run = True
                row.layer3_event_type_correct = l3.event_type_correct
                row.layer3_sentiment_correct = l3.sentiment_correct
                row.layer3_rationale_quality = l3.rationale_quality
                row.layer3_reason = l3.judge_reason

            rows.append(row)
            out_f.write(json.dumps(asdict(row)) + "\n")
            out_f.flush()
    finally:
        out_f.close()
    return rows


def aggregate(rows: list[EvalRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    l1_evaluable = [r for r in rows if "skip" not in r.layer1_reason]
    l1_pass = sum(1 for r in l1_evaluable if r.layer1_passed)

    l2_evaluable = [r for r in rows if r.layer2_return_5d is not None]
    l2_pass = sum(1 for r in l2_evaluable if r.layer2_passed)

    l3_rows = [r for r in rows if r.layer3_run]
    l3_event = sum(1 for r in l3_rows if r.layer3_event_type_correct)
    l3_sent = sum(1 for r in l3_rows if r.layer3_sentiment_correct)
    l3_quality = sum(r.layer3_rationale_quality for r in l3_rows)

    by_type_l2: dict[str, list[bool]] = defaultdict(list)
    for r in l2_evaluable:
        by_type_l2[r.classifier_event_type].append(r.layer2_passed)

    return {
        "n_total": n,
        "layer1": {
            "n_evaluable": len(l1_evaluable),
            "pass": l1_pass,
            "accuracy": l1_pass / len(l1_evaluable) if l1_evaluable else None,
        },
        "layer2": {
            "n_evaluable": len(l2_evaluable),
            "pass": l2_pass,
            "accuracy": l2_pass / len(l2_evaluable) if l2_evaluable else None,
            "by_event_type": {
                et: {"n": len(v), "accuracy": sum(v) / len(v)}
                for et, v in by_type_l2.items() if len(v) >= 5
            },
        },
        "layer3": {
            "n_evaluable": len(l3_rows),
            "event_type_accuracy": l3_event / len(l3_rows) if l3_rows else None,
            "sentiment_accuracy": l3_sent / len(l3_rows) if l3_rows else None,
            "avg_rationale_quality": l3_quality / len(l3_rows) if l3_rows else None,
        },
    }


def push_to_langsmith(rows: list[EvalRow]) -> str:
    """Create a Dataset in LangSmith with one example per evaluated filing.
    Returns the dataset URL.
    """
    client = Client()

    # Idempotent: re-create dataset if not exists
    try:
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        # delete and recreate so each run is a fresh snapshot
        client.delete_dataset(dataset_name=DATASET_NAME)
        ds = None
    except Exception:
        ds = None
    ds = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "JustTrade SEC 8-K classifier — 3-layer evaluation.\n"
            "Layer 1: rule-based event family from SEC item codes.\n"
            "Layer 2: 5-day forward return as ground truth for sentiment.\n"
            "Layer 3: Claude Opus as LLM-judge on subsample.\n"
        ),
    )

    examples_inputs = []
    examples_outputs = []
    examples_meta = []
    for r in rows:
        examples_inputs.append({
            "accession": r.accession, "ticker": r.ticker, "date_filed": r.date_filed,
            "items": r.items,
        })
        examples_outputs.append({
            "classifier_event_type": r.classifier_event_type,
            "classifier_sentiment": r.classifier_sentiment,
            "layer1_passed": r.layer1_passed,
            "layer2_passed": r.layer2_passed,
            "layer2_market_sentiment": r.layer2_market_sentiment,
            "layer3_run": r.layer3_run,
            "layer3_event_type_correct": r.layer3_event_type_correct if r.layer3_run else None,
            "layer3_sentiment_correct": r.layer3_sentiment_correct if r.layer3_run else None,
            "layer3_rationale_quality": r.layer3_rationale_quality if r.layer3_run else None,
        })
        examples_meta.append({
            "ticker": r.ticker,
            "event_type": r.classifier_event_type,
            "layer3_run": r.layer3_run,
        })

    client.create_examples(
        dataset_id=ds.id,
        inputs=examples_inputs,
        outputs=examples_outputs,
        metadata=examples_meta,
    )
    return f"https://smith.langchain.com/datasets/{ds.id}"
