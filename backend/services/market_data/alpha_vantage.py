from __future__ import annotations

import json
import logging
import os
import re
import ssl
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import certifi
from dotenv import load_dotenv


ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_PROVIDER = "alpha_vantage"
logger = logging.getLogger(__name__)
UNKNOWN_SYMBOL_VALUES = {"", "N/A", "NA", "NONE", "NULL", "-"}
NON_EQUITY_DESCRIPTION_HINTS = (
    "CASH",
    "DOLLAR",
    "GOVERNMENT OBLIG",
    "LIABILITIES",
    "OTHER ASSETS",
    "WON",
)
DESCRIPTION_SYMBOL_MAP = {
    "SAMSUNG ELECTRONICS": "SAMSUNG ELECTRONICS CO LTD",
    "MICRON TECHNOLOGY INC": "MU",
    "SANDISK CORP": "SNDK",
    "SEAGATE TECHNOLOGY HOLDINGS PLC": "STX",
    "WESTERN DIGITAL CORP": "WDC",
}
SYMBOL_NAME_MAP = {
    "MU": "MICRON TECHNOLOGY INC",
    "SNDK": "SANDISK CORP",
    "STX": "SEAGATE TECHNOLOGY HOLDINGS PLC",
    "WDC": "WESTERN DIGITAL CORP",
}


class AlphaVantageError(RuntimeError):
    pass


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_percent(value: Any) -> Optional[float]:
    text = _clean_text(value).replace("%", "").replace(",", "")
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed * 100 if 0 < parsed <= 1 else parsed


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    lower_row = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        value = lower_row.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def _normalize_symbol(value: Any) -> str:
    symbol = _clean_text(value).upper()
    return "" if symbol in UNKNOWN_SYMBOL_VALUES else symbol


def _normalize_holding_name(value: Any) -> str:
    name = _clean_text(value).upper()
    name = re.sub(r"\s*-\s*SWAP.*$", "", name)
    name = re.sub(r"\s+SWAP.*$", "", name)
    name = re.sub(r"\s+ORDINARY SHARES$", "", name)
    name = re.sub(r"\s+", " ", name).strip(" .-")
    return name


def _is_non_equity_holding(description: str) -> bool:
    normalized = description.upper()
    return any(hint in normalized for hint in NON_EQUITY_DESCRIPTION_HINTS)


def _symbol_from_description(description: str) -> str:
    normalized = _normalize_holding_name(description)
    for company_name, ticker in DESCRIPTION_SYMBOL_MAP.items():
        if normalized.startswith(company_name):
            return ticker
    return normalized


def _canonical_holding(symbol: str, description: str) -> tuple[str, str]:
    normalized_name = _normalize_holding_name(description)
    derived_symbol = _symbol_from_description(description)
    canonical_symbol = symbol or derived_symbol
    canonical_name = SYMBOL_NAME_MAP.get(canonical_symbol) or normalized_name or canonical_symbol
    return canonical_symbol, canonical_name


def parse_etf_profile_payload(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        raise AlphaVantageError(f"Alpha Vantage returned an empty response for {symbol}.")
    if payload.get("Information"):
        raise AlphaVantageError(str(payload["Information"]))
    if payload.get("Note"):
        raise AlphaVantageError(str(payload["Note"]))
    if payload.get("Error Message"):
        raise AlphaVantageError(str(payload["Error Message"]))

    holdings_payload = payload.get("holdings") or payload.get("Holdings") or []
    holdings_by_symbol: dict[str, dict[str, Any]] = {}
    skipped_rows = 0
    for row in holdings_payload:
        if not isinstance(row, dict):
            skipped_rows += 1
            continue
        description = _clean_text(_first_present(row, ["description", "name", "holding_name"]))
        constituent_symbol = _normalize_symbol(
            _first_present(row, ["symbol", "ticker", "constituent_symbol", "holding_symbol"])
        )
        weight_percent = _parse_percent(_first_present(row, ["weight", "weight_percent", "allocation"]))
        if weight_percent is None or weight_percent <= 0 or _is_non_equity_holding(description):
            skipped_rows += 1
            continue
        constituent_symbol, constituent_name = _canonical_holding(constituent_symbol, description)
        if not constituent_symbol:
            skipped_rows += 1
            continue
        holding = holdings_by_symbol.setdefault(
            constituent_symbol,
            {
                "symbol": constituent_symbol,
                "name": constituent_name,
                "weight_percent": 0.0,
                "asset_type": _clean_text(_first_present(row, ["asset_type", "type"])),
                "sector": _clean_text(_first_present(row, ["sector"])),
            },
        )
        holding["weight_percent"] += weight_percent

    logger.info(
        "Parsed Alpha Vantage ETF profile fund=%s raw_holdings=%s parsed_holdings=%s skipped_holdings=%s payload_keys=%s",
        symbol.upper(),
        len(holdings_payload),
        len(holdings_by_symbol),
        skipped_rows,
        sorted(payload.keys()),
    )

    return {
        "fund_symbol": symbol.upper(),
        "as_of_date": _clean_text(
            payload.get("as_of_date") or payload.get("asOfDate") or payload.get("latest_holding_date")
        )
        or None,
        "holdings": [
            {
                **holding,
                "weight_percent": round(holding["weight_percent"], 6),
            }
            for holding in sorted(holdings_by_symbol.values(), key=lambda item: item["weight_percent"], reverse=True)
        ],
        "raw_payload": payload,
    }


def fetch_etf_profile(symbol: str, *, api_key: Optional[str] = None, timeout: int = 20) -> dict[str, Any]:
    load_dotenv()
    key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key:
        logger.warning("Alpha Vantage API key is not configured fund=%s", symbol.upper())
        raise AlphaVantageError("ALPHA_VANTAGE_API_KEY is not configured.")

    query = urlencode({"function": "ETF_PROFILE", "symbol": symbol.upper(), "apikey": key})
    url = f"{ALPHA_VANTAGE_BASE_URL}?{query}"
    logger.info("Fetching Alpha Vantage ETF profile fund=%s has_api_key=%s", symbol.upper(), bool(key))
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urlopen(url, timeout=timeout, context=ssl_context) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise AlphaVantageError(f"Alpha Vantage HTTP error {exc.code} for {symbol}.") from exc
    except URLError as exc:
        raise AlphaVantageError(f"Could not reach Alpha Vantage for {symbol}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise AlphaVantageError(f"Alpha Vantage returned invalid JSON for {symbol}.") from exc

    logger.info(
        "Received Alpha Vantage ETF profile response fund=%s payload_keys=%s",
        symbol.upper(),
        sorted(payload.keys()) if isinstance(payload, dict) else [],
    )
    return parse_etf_profile_payload(symbol, payload)
