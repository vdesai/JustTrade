"""Pre-filter 8-K filings by item code to skip consistently-routine disclosures."""

ALWAYS_ROUTINE_ITEMS = {
    "regulation fd disclosure",
    "submission of matters to a vote of security holders",
    "amendments to articles of incorporation or bylaws",
    "amendments to articles of incorporation or bylaws; change in fiscal year",
    "change in fiscal year",
    "financial statements and exhibits",
    "shareholder director nominations",
    "shareholder nominations",
    "submission of certain matters to a vote of security holders",
    "amendment to registrant's code of ethics, or waiver of a provision of the code of ethics",
    "temporary suspension of trading under registrant's employee benefit plans",
    "change in shell company status",
}


def _normalize(item: str) -> str:
    return item.lower().strip().rstrip(".")


def is_worth_classifying(items: list[str]) -> bool:
    """True if at least one item is potentially tradable (not exclusively routine)."""
    if not items:
        return False
    normalized = {_normalize(i) for i in items}
    non_routine = normalized - ALWAYS_ROUTINE_ITEMS
    return len(non_routine) > 0
