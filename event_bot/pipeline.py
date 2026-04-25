import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from event_bot import anonymize, classifier, config, edgar, item_filter, parser, tickers


@dataclass
class ClassifiedEvent:
    cik: str
    ticker: str
    company_name: str
    date_filed: str
    accession: str
    items: list[str]
    event_type: str
    sentiment: str
    confidence: float
    tradability: float
    rationale: str


def classify_filing(
    filing: edgar.Filing,
    cik_to_ticker: dict[str, str],
    download_cache: bool = True,
    apply_item_filter: bool = True,
) -> ClassifiedEvent | None:
    try:
        raw = edgar.fetch_filing_text(filing, use_cache=download_cache)
        parsed = parser.parse_filing(raw)
        if not parsed.body_text:
            return None
        if apply_item_filter and not item_filter.is_worth_classifying(parsed.items):
            return None
        ticker = cik_to_ticker.get(filing.cik, parsed.ticker) or ""
        anon = anonymize.anonymize(parsed.body_text, company_name=filing.company_name, ticker=ticker)
        result = classifier.classify(
            anon, items=parsed.items, accession=filing.accession, ticker=ticker
        )
        return ClassifiedEvent(
            cik=filing.cik,
            ticker=ticker,
            company_name=filing.company_name,
            date_filed=filing.date_filed,
            accession=filing.accession,
            items=parsed.items,
            event_type=result.event_type,
            sentiment=result.sentiment,
            confidence=result.confidence,
            tradability=result.tradability,
            rationale=result.rationale,
        )
    except Exception as e:
        print(f"  ! {filing.accession}: {e}")
        return None


def classify_batch(
    filings: list[edgar.Filing],
    output_path: Path,
    skip_without_ticker: bool = True,
    concurrency: int = 5,
) -> list[ClassifiedEvent]:
    cik_to_ticker = tickers.load_cik_to_ticker()

    seen_accessions: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                try:
                    seen_accessions.add(json.loads(line)["accession"])
                except Exception:
                    pass
        print(f"Resuming: {len(seen_accessions)} filings already classified")

    to_process = [
        f for f in filings
        if f.accession not in seen_accessions
        and (not skip_without_ticker or f.cik in cik_to_ticker)
    ]
    print(f"Processing {len(to_process)} filings with concurrency={concurrency}")

    results: list[ClassifiedEvent] = []
    write_lock = Lock()
    f_out = open(output_path, "a")

    def worker(filing: edgar.Filing) -> ClassifiedEvent | None:
        return classify_filing(filing, cik_to_ticker)

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(worker, f): f for f in to_process}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Classifying"):
                ev = fut.result()
                if ev is None:
                    continue
                with write_lock:
                    f_out.write(json.dumps(asdict(ev)) + "\n")
                    f_out.flush()
                    results.append(ev)
    finally:
        f_out.close()

    return results


def load_classified(path: Path) -> list[ClassifiedEvent]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(ClassifiedEvent(**d))
    return out
