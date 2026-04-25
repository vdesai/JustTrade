"""Three independent evaluation layers for the SEC filing classifier.

Layer 1 — Event family smoke test (cheap, weak)
    Source:  filing item codes (header)
    Method:  deterministic regex + lookup table
    Catches: gross hallucinations (classifier output completely off-script)

Layer 2 — Market-derived sentiment (independent, strong)
    Source:  5-day forward stock return (data Claude could not have seen)
    Method:  threshold on return magnitude
    Catches: whether Claude's read of the filing actually predicts the stock

Layer 3 — LLM-as-judge with a different model (medium)
    Source:  same filing, but graded by Claude Opus instead of Haiku classifier
    Method:  ask judge model to score the classification on a rubric
    Catches: subtle subtype errors, rationale quality

The three layers triangulate. None alone is sufficient.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import anthropic
from dotenv import load_dotenv
from langsmith import traceable

from event_bot import config, prices

load_dotenv(config.ROOT / ".env")


# =============================================================================
# Layer 1 — Event family from item codes
# =============================================================================

ITEM_TO_FAMILY: dict[str, str] = {
    # earnings
    "results of operations": "earnings",
    "results of operations and financial condition": "earnings",
    # M&A
    "completion of acquisition": "m_and_a",
    "completion of acquisition or disposition of assets": "m_and_a",
    "entry into a material definitive agreement": "agreement_or_m_and_a",
    "termination of a material definitive agreement": "agreement_or_m_and_a",
    # exec changes
    "departure of directors or certain officers": "exec_change",
    "departure of directors or certain officers; election of directors; appointment of certain officers: compensatory arrangements of certain officers": "exec_change",
    "amendments to articles of incorporation or bylaws": "governance",
    # capital structure
    "unregistered sales of equity securities": "stock_offering",
    "material modification to rights of security holders": "capital_structure",
    "creation of a direct financial obligation": "debt_or_obligation",
    "creation of a direct financial obligation or an obligation under an off-balance sheet arrangement": "debt_or_obligation",
    "triggering events that accelerate or increase a direct financial obligation or an obligation under an off-balance sheet arrangement": "debt_or_obligation",
    "costs associated with exit or disposal activities": "restructuring",
    "material impairments": "impairment_or_restructuring",
    # bankruptcy / going concern
    "bankruptcy or receivership": "bankruptcy",
    # regulatory / legal
    "notice of delisting or failure to satisfy a continued listing rule or standard": "delisting",
    "non-reliance on previously issued financial statements or a related audit report or completed interim review": "non_reliance",
    "changes in registrant's certifying accountant": "auditor_change",
    # market/news catch-alls (could be many things)
    "regulation fd disclosure": "fd_disclosure",
    "other events": "other_or_news",  # 8.01 — most variable
    "submission of matters to a vote of security holders": "shareholder_vote",
    "amendment to registrant's code of ethics": "governance",
    "shareholder director nominations": "governance",
}


# Map specific event_type values from the classifier to the family they should belong to.
EVENT_TYPE_TO_FAMILY: dict[str, list[str]] = {
    "earnings_beat":          ["earnings"],
    "earnings_miss":          ["earnings"],
    "earnings_inline":        ["earnings"],
    "guidance_raised":        ["earnings", "other_or_news"],
    "guidance_lowered":       ["earnings", "other_or_news"],
    "guidance_reaffirmed":    ["earnings", "other_or_news"],
    "m_and_a_target":         ["m_and_a", "agreement_or_m_and_a"],
    "m_and_a_acquirer":       ["m_and_a", "agreement_or_m_and_a"],
    "material_agreement":     ["agreement_or_m_and_a"],
    "exec_departure":         ["exec_change"],
    "exec_appointment":       ["exec_change"],
    "dividend_initiated_or_raised": ["other_or_news", "earnings"],
    "dividend_cut":           ["other_or_news", "earnings"],
    "buyback_announced":      ["other_or_news", "earnings", "capital_structure"],
    "stock_offering":         ["stock_offering", "capital_structure"],
    "debt_issuance":          ["debt_or_obligation", "capital_structure"],
    "fda_approval":           ["other_or_news"],
    "fda_rejection":          ["other_or_news"],
    "clinical_trial_positive": ["other_or_news"],
    "clinical_trial_negative": ["other_or_news"],
    "legal_lawsuit_filed":    ["other_or_news"],
    "legal_settlement":       ["other_or_news", "agreement_or_m_and_a"],
    "restructuring_layoffs":  ["restructuring", "impairment_or_restructuring", "other_or_news"],
    "cyber_incident":         ["other_or_news"],
    "bankruptcy":             ["bankruptcy"],
    "going_concern":          ["other_or_news", "non_reliance", "impairment_or_restructuring", "delisting"],
    "product_launch":         ["other_or_news"],
    "contract_win":           ["other_or_news", "agreement_or_m_and_a"],
    "other":                  list(set(sum(([f] for f in ITEM_TO_FAMILY.values()), []))),
}


def _families_from_items(items: list[str]) -> set[str]:
    """Return the set of plausible families given the filing's item codes."""
    out: set[str] = set()
    for item in items:
        key = item.strip().lower().rstrip(".")
        if key in ITEM_TO_FAMILY:
            out.add(ITEM_TO_FAMILY[key])
    return out


@dataclass
class Layer1Result:
    passed: bool
    classifier_event_type: str
    item_families: list[str]
    expected_families: list[str]
    reason: str


def grade_layer1_smoke_test(
    classifier_event_type: str,
    items: list[str],
) -> Layer1Result:
    """Pass if the classifier's event_type belongs to one of the families
    derivable from the filing's item codes.
    """
    item_fams = _families_from_items(items)
    expected = set(EVENT_TYPE_TO_FAMILY.get(classifier_event_type, []))
    if not item_fams:
        return Layer1Result(
            passed=True,
            classifier_event_type=classifier_event_type,
            item_families=[], expected_families=list(expected),
            reason="no item codes parsed; cannot evaluate (skip)",
        )
    overlap = item_fams & expected
    if overlap:
        return Layer1Result(
            passed=True,
            classifier_event_type=classifier_event_type,
            item_families=list(item_fams), expected_families=list(expected),
            reason=f"event_type family ∈ filing families ({list(overlap)[0]})",
        )
    return Layer1Result(
        passed=False,
        classifier_event_type=classifier_event_type,
        item_families=list(item_fams), expected_families=list(expected),
        reason=f"event_type='{classifier_event_type}' not in any item family {list(item_fams)}",
    )


# =============================================================================
# Layer 2 — Market-derived sentiment ground truth
# =============================================================================

# Returns over 5 trading days are noisy; use 2% threshold to label.
BULLISH_THRESHOLD = 0.02
BEARISH_THRESHOLD = -0.02


@dataclass
class Layer2Result:
    passed: bool
    classifier_sentiment: str
    market_sentiment: str
    return_5d: float | None
    reason: str


def grade_layer2_market_truth(
    ticker: str,
    date_filed: str,
    classifier_sentiment: str,
) -> Layer2Result:
    """Pass if classifier's sentiment matches the direction the stock actually moved
    over the 5 trading days following the filing.
    """
    try:
        rets = prices.returns_around_event(ticker, date_filed)
    except Exception as e:
        return Layer2Result(
            passed=True,  # skip when no data — not a failure of classifier
            classifier_sentiment=classifier_sentiment,
            market_sentiment="unknown", return_5d=None,
            reason=f"price fetch failed: {e}",
        )
    if not rets or "ret_5d" not in rets:
        return Layer2Result(
            passed=True,
            classifier_sentiment=classifier_sentiment,
            market_sentiment="unknown", return_5d=None,
            reason="no 5d return available",
        )
    r5 = rets["ret_5d"]
    if r5 > BULLISH_THRESHOLD:
        market = "bullish"
    elif r5 < BEARISH_THRESHOLD:
        market = "bearish"
    else:
        market = "neutral"

    if classifier_sentiment == "neutral":
        # When classifier says neutral, count as pass if the actual move stayed within thresholds.
        passed = market == "neutral"
    else:
        passed = classifier_sentiment == market
    return Layer2Result(
        passed=passed,
        classifier_sentiment=classifier_sentiment,
        market_sentiment=market,
        return_5d=r5,
        reason=f"classifier='{classifier_sentiment}' vs market='{market}' (ret_5d={r5*100:+.2f}%)",
    )


# =============================================================================
# Layer 3 — LLM-as-judge using a different model
# =============================================================================

JUDGE_MODEL = "claude-opus-4-7"  # Opus judges Haiku — different model, different blind spots
JUDGE_MAX_TOKENS = 256

JUDGE_PROMPT = """You are grading a classification of an anonymized SEC 8-K filing.

A different system classified this filing. Read the filing and judge whether
its classification is correct.

Filing (anonymized):
---
{body}
---

Items disclosed: {items}

The classification under review:
  event_type: {event_type}
  sentiment: {sentiment}
  rationale: {rationale}

Output ONLY a JSON object:
{{
  "event_type_correct": true | false,
  "sentiment_correct": true | false,
  "rationale_quality": integer 1-5 (5=excellent, 1=incoherent),
  "judge_reason": "<= 25 words on what's right or wrong"
}}"""


@dataclass
class Layer3Result:
    event_type_correct: bool
    sentiment_correct: bool
    rationale_quality: int
    judge_reason: str
    error: str = ""


_judge_client: anthropic.Anthropic | None = None


def _get_judge_client() -> anthropic.Anthropic:
    global _judge_client
    if _judge_client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        _judge_client = anthropic.Anthropic(api_key=key)
    return _judge_client


@traceable(run_type="chain", name="layer3_llm_judge")
def grade_layer3_llm_judge(
    anonymized_body: str,
    items: list[str],
    classifier_event_type: str,
    classifier_sentiment: str,
    classifier_rationale: str,
    accession: str | None = None,
) -> Layer3Result:
    body = anonymized_body[:6000]
    prompt = JUDGE_PROMPT.format(
        body=body,
        items="; ".join(items),
        event_type=classifier_event_type,
        sentiment=classifier_sentiment,
        rationale=classifier_rationale[:300],
    )
    try:
        client = _get_judge_client()
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return Layer3Result(False, False, 0, "no_json", error="parse")
        data = json.loads(m.group(0))
        return Layer3Result(
            event_type_correct=bool(data.get("event_type_correct", False)),
            sentiment_correct=bool(data.get("sentiment_correct", False)),
            rationale_quality=int(data.get("rationale_quality", 0)),
            judge_reason=str(data.get("judge_reason", ""))[:200],
        )
    except Exception as e:
        return Layer3Result(False, False, 0, "judge_error", error=str(e)[:200])
