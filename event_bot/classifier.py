import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Optional

import anthropic
from dotenv import load_dotenv

from event_bot import config

load_dotenv(config.ROOT / ".env")

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key or key.startswith("sk-ant-..."):
            raise RuntimeError("ANTHROPIC_API_KEY missing or placeholder in .env")
        _client = anthropic.Anthropic(api_key=key)
    return _client


SYSTEM_PROMPT = """You classify anonymized SEC 8-K filings for an event-driven trading system.

The filing is ALWAYS anonymized: company names replaced with [COMPANY], tickers with [TICKER], dates with [DATE] or [YEAR]. Do NOT attempt to guess the company or ticker — the goal is to classify the EVENT, not the issuer.

For each filing, output ONLY a JSON object (no prose) with these fields:

{
  "event_type": one of [
    "earnings_beat",
    "earnings_miss",
    "earnings_inline",
    "guidance_raised",
    "guidance_lowered",
    "guidance_reaffirmed",
    "m_and_a_target",
    "m_and_a_acquirer",
    "material_agreement",
    "exec_departure",
    "exec_appointment",
    "dividend_initiated_or_raised",
    "dividend_cut",
    "buyback_announced",
    "stock_offering",
    "debt_issuance",
    "fda_approval",
    "fda_rejection",
    "clinical_trial_positive",
    "clinical_trial_negative",
    "legal_lawsuit_filed",
    "legal_settlement",
    "restructuring_layoffs",
    "cyber_incident",
    "bankruptcy",
    "going_concern",
    "product_launch",
    "contract_win",
    "other"
  ],
  "sentiment": one of ["bullish", "neutral", "bearish"] (from the perspective of the stock's likely short-term price response),
  "confidence": float in [0, 1] (how confident you are in this classification),
  "tradability": float in [0, 1] (how likely this is to cause a tradable short-term price move, independent of direction),
  "rationale": string of <= 25 words explaining your classification
}

Be conservative: if the filing is boilerplate, routine, or ambiguous, use "other" with low tradability."""


@dataclass
class Classification:
    event_type: str
    sentiment: str
    confidence: float
    tradability: float
    rationale: str
    raw_response: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(text: str) -> Classification:
    m = _JSON_RE.search(text)
    if not m:
        return Classification(
            event_type="other", sentiment="neutral", confidence=0.0,
            tradability=0.0, rationale="no JSON in response",
            raw_response=text, error="no_json",
        )
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return Classification(
            event_type="other", sentiment="neutral", confidence=0.0,
            tradability=0.0, rationale=f"json parse error",
            raw_response=text, error=f"json_error:{e}",
        )
    return Classification(
        event_type=str(data.get("event_type", "other")),
        sentiment=str(data.get("sentiment", "neutral")),
        confidence=float(data.get("confidence", 0.0)),
        tradability=float(data.get("tradability", 0.0)),
        rationale=str(data.get("rationale", ""))[:200],
        raw_response=text,
    )


def classify(anonymized_body: str, items: list[str] | None = None) -> Classification:
    body = anonymized_body[: config.CLASSIFIER_BODY_CHAR_LIMIT]
    item_header = ""
    if items:
        item_header = "SEC 8-K items disclosed: " + "; ".join(items) + "\n\n"
    user_content = f"{item_header}Filing body (anonymized):\n---\n{body}\n---"

    client = _get_client()
    resp = client.messages.create(
        model=config.CLASSIFIER_MODEL,
        max_tokens=config.CLASSIFIER_MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    return _parse_response(text)
