"""Deterministic field normalization. Never invents missing characters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class NormalizedField:
    field_type: str
    raw: str
    value: Any
    valid: bool
    uncertain: bool
    details: dict[str, Any]


_AMOUNT_RE = re.compile(
    r"(?P<number>(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d{1,2})?)\s*(?P<currency>TL|TRY|₺|USD|\$|EUR|€)?",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
_PLATE_RE = re.compile(r"\b(\d{2})\s*([A-ZÇĞİÖŞÜ]{1,3})\s*(\d{2,4})\b", re.IGNORECASE)


def normalize_amount(raw: str) -> NormalizedField:
    match = _AMOUNT_RE.search(raw.replace(" ", ""))
    if not match:
        return NormalizedField("amount", raw, None, False, True, {"reason": "unparsed"})
    number = match.group("number")
    currency_raw = (match.group("currency") or "TRY").upper().replace("₺", "TRY").replace("$", "USD").replace("€", "EUR")
    if currency_raw == "TL":
        currency_raw = "TRY"
    normalized_number = number.replace(".", "").replace(",", ".")
    try:
        value = f"{float(normalized_number):.2f}"
    except ValueError:
        return NormalizedField("amount", raw, None, False, True, {"reason": "invalid_number"})
    return NormalizedField("amount", raw, {"amount": value, "currency": currency_raw}, True, False, {})


def normalize_date(raw: str) -> NormalizedField:
    match = _DATE_RE.search(raw)
    if not match:
        return NormalizedField("date", raw, None, False, True, {"reason": "unparsed"})
    day, month, year = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return NormalizedField("date", raw, None, False, True, {"reason": "invalid_date"})
    return NormalizedField("date", raw, f"{year:04d}-{month:02d}-{day:02d}", True, False, {})


def normalize_plate(raw: str) -> NormalizedField:
    match = _PLATE_RE.search(raw)
    if not match:
        return NormalizedField("plate", raw, raw.strip(), False, True, {"reason": "unparsed"})
    value = f"{int(match.group(1)):02d} {match.group(2).upper()} {match.group(3)}"
    return NormalizedField("plate", raw, value, True, False, {})


def normalize_document_number(raw: str) -> NormalizedField:
    value = raw.strip()
    if not value:
        return NormalizedField("document_number", raw, None, False, True, {"reason": "empty"})
    return NormalizedField("document_number", raw, value, True, False, {"kept_leading_zeros": True})


def normalize_field(field_type: str, raw: str) -> NormalizedField:
    mapping = {
        "amount": normalize_amount,
        "date": normalize_date,
        "plate": normalize_plate,
        "document_number": normalize_document_number,
    }
    if field_type not in mapping:
        return NormalizedField(field_type, raw, raw, True, False, {})
    return mapping[field_type](raw)
