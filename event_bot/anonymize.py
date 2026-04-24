import re

_GENERIC_SUFFIXES = {
    "INC", "INC.", "CORP", "CORP.", "CORPORATION", "CO", "CO.", "COMPANY",
    "LTD", "LTD.", "LIMITED", "LLC", "L.L.C.", "PLC", "TRUST", "HOLDINGS",
    "GROUP", "PARTNERS", "LP", "L.P.", "SA", "S.A.", "NV", "N.V.", "AG",
    "THE", "&",
}

_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:,?\s+\d{4})?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMERIC_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")

_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")


def _name_variants(company_name: str) -> list[str]:
    if not company_name:
        return []
    name = re.sub(r"[^\w\s&.]", " ", company_name).strip()
    tokens = name.split()
    significant = [t for t in tokens if t.upper().strip(".,") not in _GENERIC_SUFFIXES]

    variants: set[str] = {company_name, company_name.upper(), company_name.title()}
    if significant:
        full_sig = " ".join(significant)
        variants.update({full_sig, full_sig.upper(), full_sig.title()})
        if len(significant) >= 1 and len(significant[0]) >= 4:
            variants.add(significant[0])
            variants.add(significant[0].upper())
            variants.add(significant[0].title())
    return sorted(variants, key=len, reverse=True)


def anonymize(text: str, company_name: str = "", ticker: str = "") -> str:
    if not text:
        return text

    text = _URL_RE.sub("[URL]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)

    if ticker:
        for t in {ticker, ticker.upper(), ticker.lower(), f"${ticker.upper()}"}:
            text = re.sub(rf"\b{re.escape(t)}\b", "[TICKER]", text)

    for variant in _name_variants(company_name):
        if len(variant) < 3:
            continue
        text = re.sub(re.escape(variant), "[COMPANY]", text, flags=re.IGNORECASE)

    text = _DATE_RE.sub("[DATE]", text)
    text = _NUMERIC_DATE_RE.sub("[DATE]", text)
    text = _YEAR_RE.sub("[YEAR]", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text
