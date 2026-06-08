from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable, Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.services.market_data.alpha_vantage import ALPHA_VANTAGE_PROVIDER, AlphaVantageError, fetch_etf_profile
from backend.utils.investments_db import get_investments_connection

logger = logging.getLogger(__name__)


FIDELITY_POSITION_HEADERS = {
    "Account Number",
    "Account Name",
    "Symbol",
    "Description",
    "Current Value",
}
FIXED_INCOME_HINTS = ("TREASURY", "TREASRY", "BOND", "T-BILL", "0-3 MNTH")
FUND_HINTS = (" ETF", " ETF ", "INDEX FUND", "ISHARES", "VANGUARD", "INVESCO", "ROUNDHILL", " TR ")
DIRECT_STOCK_HINTS = (" COM", " COMMON STOCK", " INC", " CORP", " LTD", " PLC")
INDUSTRY_CLASSIFICATION_PROVIDER = "openai"
INDUSTRY_FALLBACK_PROVIDER = "deterministic_fallback"
INDUSTRY_CLASSIFIER_SYSTEM_PROMPT = """
You classify public companies into investment sectors and industries.
Return valid JSON only with key classifications.
Each classification must include:
- symbol
- name
- sector
- industry
- confidence, a number from 0 to 1
- rationale, one short phrase

Rules:
- Use standard, investor-friendly labels.
- Prefer specific industries such as Semiconductors, Consumer Electronics, Software, Internet Services, E-commerce, Search & Advertising, Social Media, Cloud Infrastructure.
- Do not invent companies. Return one classification for each input item.
- If a company is unfamiliar, use sector "Unknown", industry "Unknown", confidence below 0.5.
""".strip()
INDUSTRY_FALLBACK_RULES = [
    (("NVIDIA", "NVDA", "MICRON", "MU", "BROADCOM", "AVGO", "AMD", "SK HYNIX", "SAMSUNG ELECTRONICS", "SEMICONDUCTOR", "SEAGATE", "WESTERN DIGITAL", "SANDISK"), "Information Technology", "Semiconductors"),
    (("APPLE", "AAPL"), "Information Technology", "Consumer Electronics"),
    (("MICROSOFT", "MSFT"), "Information Technology", "Software"),
    (("AMAZON", "AMZN"), "Consumer Discretionary", "E-commerce & Cloud"),
    (("GOOGLE", "ALPHABET", "GOOG", "GOOGL"), "Communication Services", "Search & Advertising"),
    (("META", "FACEBOOK"), "Communication Services", "Social Media"),
    (("TESLA", "TSLA"), "Consumer Discretionary", "Electric Vehicles"),
]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clean_text(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _parse_number(value: Any) -> Optional[float]:
    text = _clean_text(value)
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace("$", "").replace(",", "").replace("%", "").replace("+", "")
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return -parsed if negative else parsed


def _round_money(value: Optional[float]) -> Optional[float]:
    return round(float(value), 2) if value is not None else None


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_as_of_date(value: Optional[str]) -> Optional[str]:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text.replace("/", "-")).isoformat()
    except ValueError as exc:
        raise ValueError("as_of_date must use YYYY-MM-DD format.") from exc


def _normalize_symbol(value: Any) -> str:
    return _clean_text(value).upper()


def _is_fixed_income_like(symbol: str, description: str) -> bool:
    normalized = f"{symbol} {description}".upper()
    return any(hint in normalized for hint in FIXED_INCOME_HINTS)


def _is_fund_candidate(symbol: str, description: str, *, is_cash_like: bool = False) -> bool:
    if is_cash_like or _is_fixed_income_like(symbol, description):
        return False
    normalized = f" {description.upper()} "
    if any(hint in normalized for hint in FUND_HINTS):
        return True
    return not any(hint in normalized for hint in DIRECT_STOCK_HINTS)


def _is_cash_like(symbol: str, description: str, quantity: Optional[float], last_price: Optional[float]) -> bool:
    normalized_description = description.upper()
    return (
        symbol.endswith("**")
        or "MONEY MARKET" in normalized_description
        or "HELD IN CASH" in normalized_description
        or (quantity is None and last_price is None)
    )


def _decode_csv_payload(csv_payload: bytes | str) -> str:
    if isinstance(csv_payload, bytes):
        return csv_payload.decode("utf-8-sig")
    return csv_payload.lstrip("\ufeff")


def _iter_fidelity_position_rows(csv_payload: bytes | str) -> Iterable[dict[str, Any]]:
    text = _decode_csv_payload(csv_payload)
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not FIDELITY_POSITION_HEADERS <= set(reader.fieldnames):
        raise ValueError("CSV does not look like a Fidelity positions export.")

    for row in reader:
        account_number = _clean_text(row.get("Account Number"))
        account_name = _clean_text(row.get("Account Name"))
        symbol = _normalize_symbol(row.get("Symbol"))
        current_value = _parse_number(row.get("Current Value"))

        # Fidelity appends legal/footer rows after the position rows. They do
        # not have the required position shape, so skip them.
        if not account_number or not account_name or not symbol or current_value is None:
            continue

        description = _clean_text(row.get("Description"))
        quantity = _parse_number(row.get("Quantity"))
        last_price = _parse_number(row.get("Last Price"))
        yield {
            "account_number": account_number,
            "account_name": account_name,
            "symbol": symbol,
            "description": description,
            "quantity": quantity,
            "last_price": last_price,
            "last_price_change": _parse_number(row.get("Last Price Change")),
            "current_value": current_value,
            "day_gain_loss_amount": _parse_number(row.get("Today's Gain/Loss Dollar")),
            "day_gain_loss_percent": _parse_number(row.get("Today's Gain/Loss Percent")),
            "total_gain_loss_amount": _parse_number(row.get("Total Gain/Loss Dollar")),
            "total_gain_loss_percent": _parse_number(row.get("Total Gain/Loss Percent")),
            "percent_of_account": _parse_number(row.get("Percent Of Account")),
            "cost_basis_total": _parse_number(row.get("Cost Basis Total")),
            "average_cost_basis": _parse_number(row.get("Average Cost Basis")),
            "fidelity_type": _clean_text(row.get("Type")),
            "is_cash_like": _is_cash_like(symbol, description, quantity, last_price),
        }


def _account_id_for(account_number: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fidelity:{account_number}"))


def import_fidelity_positions_csv(
    csv_payload: bytes | str,
    *,
    filename: Optional[str] = None,
    as_of_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    rows = list(_iter_fidelity_position_rows(csv_payload))
    if not rows:
        raise ValueError("No Fidelity position rows were found in the CSV.")

    normalized_as_of_date = _normalize_as_of_date(as_of_date)
    import_id = str(uuid.uuid4())
    imported_at = _now_iso()

    with get_investments_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO investment_imports (import_id, source, filename, as_of_date, imported_at, row_count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (import_id, "fidelity_positions_csv", filename, normalized_as_of_date, imported_at, len(rows), "completed"),
        )

        account_ids: dict[str, str] = {}
        for row in rows:
            account_id = _account_id_for(row["account_number"])
            account_ids[row["account_number"]] = account_id
            conn.execute(
                """
                INSERT INTO investment_accounts (
                    account_id, broker, external_account_id, account_name, display_name, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(broker, external_account_id)
                DO UPDATE SET
                    account_name = excluded.account_name,
                    display_name = excluded.display_name,
                    updated_at = excluded.updated_at
                """,
                (
                    account_id,
                    "Fidelity",
                    row["account_number"],
                    row["account_name"],
                    row["account_name"],
                    imported_at,
                    imported_at,
                ),
            )

        conn.executemany(
            """
            INSERT INTO investment_positions (
                import_id, account_id, symbol, description, quantity, last_price, last_price_change,
                current_value, day_gain_loss_amount, day_gain_loss_percent, total_gain_loss_amount,
                total_gain_loss_percent, percent_of_account, cost_basis_total, average_cost_basis,
                fidelity_type, is_cash_like, imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    import_id,
                    account_ids[row["account_number"]],
                    row["symbol"],
                    row["description"],
                    row["quantity"],
                    row["last_price"],
                    row["last_price_change"],
                    row["current_value"],
                    row["day_gain_loss_amount"],
                    row["day_gain_loss_percent"],
                    row["total_gain_loss_amount"],
                    row["total_gain_loss_percent"],
                    row["percent_of_account"],
                    row["cost_basis_total"],
                    row["average_cost_basis"],
                    row["fidelity_type"],
                    1 if row["is_cash_like"] else 0,
                    imported_at,
                )
                for row in rows
            ],
        )
        conn.commit()

    return {
        "import_id": import_id,
        "source": "fidelity_positions_csv",
        "filename": filename,
        "as_of_date": normalized_as_of_date,
        "imported_at": imported_at,
        "row_count": len(rows),
        "account_count": len(account_ids),
        "total_value": _round_money(sum(row["current_value"] for row in rows)) or 0.0,
        "status": "completed",
    }


def _latest_import(conn) -> Optional[dict[str, Any]]:
    row = conn.execute(
        """
        SELECT import_id, source, filename, as_of_date, imported_at, row_count, status
        FROM investment_imports
        WHERE status = 'completed'
        ORDER BY imported_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    latest_import = dict(row)
    latest_import["as_of_date"] = _normalize_as_of_date(latest_import.get("as_of_date"))
    return latest_import


def get_investments_overview(*, db_path: Optional[str] = None) -> dict[str, Any]:
    with get_investments_connection(db_path) as conn:
        latest_import = _latest_import(conn)
        if latest_import is None:
            return {
                "status": "empty",
                "latest_import": None,
                "summary": {
                    "total_value": 0.0,
                    "cash_value": 0.0,
                    "holding_count": 0,
                    "account_count": 0,
                    "day_gain_loss_amount": 0.0,
                    "total_gain_loss_amount": 0.0,
                    "cost_basis_total": 0.0,
                },
                "accounts": [],
                "holdings": [],
            }

        positions = conn.execute(
            """
            SELECT
                p.position_id,
                p.symbol,
                p.description,
                p.quantity,
                p.last_price,
                p.last_price_change,
                p.current_value,
                p.day_gain_loss_amount,
                p.day_gain_loss_percent,
                p.total_gain_loss_amount,
                p.total_gain_loss_percent,
                p.percent_of_account,
                p.cost_basis_total,
                p.average_cost_basis,
                p.fidelity_type,
                p.is_cash_like,
                a.account_id,
                a.account_name,
                a.display_name,
                a.external_account_id
            FROM investment_positions p
            JOIN investment_accounts a ON a.account_id = p.account_id
            WHERE p.import_id = ?
            ORDER BY p.current_value DESC, p.symbol ASC
            """,
            (latest_import["import_id"],),
        ).fetchall()

    holdings = [
        {
            "position_id": row["position_id"],
            "account_id": row["account_id"],
            "account_name": row["account_name"],
            "symbol": row["symbol"],
            "description": row["description"],
            "quantity": row["quantity"],
            "last_price": _round_money(row["last_price"]),
            "last_price_change": _round_money(row["last_price_change"]),
            "current_value": _round_money(row["current_value"]) or 0.0,
            "day_gain_loss_amount": _round_money(row["day_gain_loss_amount"]),
            "day_gain_loss_percent": row["day_gain_loss_percent"],
            "total_gain_loss_amount": _round_money(row["total_gain_loss_amount"]),
            "total_gain_loss_percent": row["total_gain_loss_percent"],
            "percent_of_account": row["percent_of_account"],
            "cost_basis_total": _round_money(row["cost_basis_total"]),
            "average_cost_basis": _round_money(row["average_cost_basis"]),
            "fidelity_type": row["fidelity_type"],
            "is_cash_like": bool(row["is_cash_like"]),
        }
        for row in positions
    ]

    account_totals: dict[str, dict[str, Any]] = {}
    for holding in holdings:
        account = account_totals.setdefault(
            holding["account_id"],
            {
                "account_id": holding["account_id"],
                "account_name": holding["account_name"],
                "total_value": 0.0,
                "cash_value": 0.0,
                "holding_count": 0,
            },
        )
        account["total_value"] += holding["current_value"]
        account["cash_value"] += holding["current_value"] if holding["is_cash_like"] else 0.0
        account["holding_count"] += 1

    total_value = sum(holding["current_value"] for holding in holdings)
    cash_value = sum(holding["current_value"] for holding in holdings if holding["is_cash_like"])
    day_gain = sum(holding["day_gain_loss_amount"] or 0.0 for holding in holdings)
    total_gain = sum(holding["total_gain_loss_amount"] or 0.0 for holding in holdings)
    cost_basis = sum(holding["cost_basis_total"] or 0.0 for holding in holdings)
    for holding in holdings:
        holding["percent_of_portfolio"] = round((holding["current_value"] / total_value) * 100, 2) if total_value else 0.0

    return {
        "status": "ready",
        "latest_import": latest_import,
        "summary": {
            "total_value": _round_money(total_value) or 0.0,
            "cash_value": _round_money(cash_value) or 0.0,
            "holding_count": len(holdings),
            "account_count": len(account_totals),
            "day_gain_loss_amount": _round_money(day_gain) or 0.0,
            "day_gain_loss_percent": round((day_gain / total_value) * 100, 2) if total_value else 0.0,
            "total_gain_loss_amount": _round_money(total_gain) or 0.0,
            "total_gain_loss_percent": round((total_gain / cost_basis) * 100, 2) if cost_basis else 0.0,
            "cost_basis_total": _round_money(cost_basis) or 0.0,
        },
        "accounts": [
            {
                **account,
                "total_value": _round_money(account["total_value"]) or 0.0,
                "cash_value": _round_money(account["cash_value"]) or 0.0,
            }
            for account in sorted(account_totals.values(), key=lambda item: item["total_value"], reverse=True)
        ],
        "holdings": holdings,
    }


def _latest_positions(conn) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    latest_import = _latest_import(conn)
    if latest_import is None:
        return None, []

    rows = conn.execute(
        """
        SELECT
            p.position_id,
            p.symbol,
            p.description,
            p.current_value,
            p.is_cash_like,
            a.account_id,
            a.account_name
        FROM investment_positions p
        JOIN investment_accounts a ON a.account_id = p.account_id
        WHERE p.import_id = ?
        ORDER BY p.current_value DESC, p.symbol ASC
        """,
        (latest_import["import_id"],),
    ).fetchall()
    return latest_import, [dict(row) for row in rows]


def _latest_completed_fund_snapshot(conn, fund_symbol: str, provider: str = ALPHA_VANTAGE_PROVIDER) -> Optional[dict[str, Any]]:
    row = conn.execute(
        """
        SELECT snapshot_id, fund_symbol, provider, as_of_date, fetched_at, status, error_message
        FROM investment_fund_snapshots
        WHERE fund_symbol = ? AND provider = ? AND status = 'completed'
        ORDER BY fetched_at DESC
        LIMIT 1
        """,
        (fund_symbol.upper(), provider),
    ).fetchone()
    return dict(row) if row else None


def _snapshot_is_fresh(snapshot: Optional[dict[str, Any]], *, max_age_hours: int) -> bool:
    fetched_at = _parse_iso_datetime(snapshot.get("fetched_at") if snapshot else None)
    if fetched_at is None:
        return False
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=UTC)
    return datetime.now(UTC) - fetched_at < timedelta(hours=max_age_hours)


def _insert_fund_snapshot(
    conn,
    *,
    fund_symbol: str,
    provider: str,
    as_of_date: Optional[str],
    status: str,
    holdings: list[dict[str, Any]],
    raw_payload: Optional[dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> str:
    snapshot_id = str(uuid.uuid4())
    fetched_at = _now_iso()
    conn.execute(
        """
        INSERT INTO investment_fund_snapshots (
            snapshot_id, fund_symbol, provider, as_of_date, fetched_at, status, error_message, raw_payload_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot_id,
            fund_symbol.upper(),
            provider,
            _normalize_as_of_date(as_of_date) if as_of_date else None,
            fetched_at,
            status,
            error_message,
            json.dumps(raw_payload or {}, sort_keys=True),
        ),
    )
    conn.executemany(
        """
        INSERT INTO investment_fund_holdings (
            snapshot_id, fund_symbol, constituent_symbol, constituent_name, weight_percent, asset_type, sector
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                snapshot_id,
                fund_symbol.upper(),
                holding["symbol"],
                holding.get("name") or "",
                float(holding["weight_percent"]),
                holding.get("asset_type") or "",
                holding.get("sector") or "",
            )
            for holding in holdings
        ],
    )
    return snapshot_id


def _fund_candidates_from_positions(positions: list[dict[str, Any]]) -> list[str]:
    symbols = {
        row["symbol"]
        for row in positions
        if row.get("current_value", 0) > 0
        and _is_fund_candidate(row["symbol"], row.get("description") or "", is_cash_like=bool(row.get("is_cash_like")))
    }
    return sorted(symbols)


def refresh_investment_fund_holdings(
    *,
    db_path: Optional[str] = None,
    force: bool = False,
    max_age_hours: int = 24,
    request_delay_seconds: float = 1.2,
    profile_fetcher: Callable[[str], dict[str, Any]] = fetch_etf_profile,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    logger.info(
        "Starting investment fund holdings refresh force=%s max_age_hours=%s request_delay_seconds=%s",
        force,
        max_age_hours,
        request_delay_seconds,
    )
    with get_investments_connection(db_path) as conn:
        latest_import, positions = _latest_positions(conn)
        if latest_import is None:
            logger.info("Investment fund holdings refresh skipped: no investment snapshot found")
            return {"status": "empty", "refreshed": [], "skipped": [], "failed": [], "fund_candidates": []}

        fund_symbols = _fund_candidates_from_positions(positions)
        logger.info(
            "Investment fund holdings refresh candidates import_id=%s positions=%s candidates=%s",
            latest_import.get("import_id"),
            len(positions),
            fund_symbols,
        )
        refreshed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        provider_request_count = 0

        for symbol in fund_symbols:
            cached_snapshot = _latest_completed_fund_snapshot(conn, symbol)
            if not force and _snapshot_is_fresh(cached_snapshot, max_age_hours=max_age_hours):
                logger.info(
                    "Skipping Alpha Vantage refresh for %s: fresh cache fetched_at=%s",
                    symbol,
                    cached_snapshot["fetched_at"],
                )
                skipped.append({"symbol": symbol, "reason": "fresh_cache", "fetched_at": cached_snapshot["fetched_at"]})
                continue

            try:
                if provider_request_count > 0 and request_delay_seconds > 0:
                    logger.info(
                        "Waiting %.2fs before next Alpha Vantage request to avoid free-tier burst limit",
                        request_delay_seconds,
                    )
                    sleeper(request_delay_seconds)
                provider_request_count += 1
                logger.info("Refreshing Alpha Vantage holdings for %s", symbol)
                profile = profile_fetcher(symbol)
                snapshot_id = _insert_fund_snapshot(
                    conn,
                    fund_symbol=symbol,
                    provider=ALPHA_VANTAGE_PROVIDER,
                    as_of_date=profile.get("as_of_date"),
                    status="completed",
                    holdings=profile.get("holdings") or [],
                    raw_payload=profile.get("raw_payload") or {},
                )
                refreshed.append(
                    {
                        "symbol": symbol,
                        "snapshot_id": snapshot_id,
                        "holding_count": len(profile.get("holdings") or []),
                    }
                )
                logger.info(
                    "Stored Alpha Vantage holdings for %s snapshot_id=%s holding_count=%s as_of_date=%s",
                    symbol,
                    snapshot_id,
                    len(profile.get("holdings") or []),
                    profile.get("as_of_date"),
                )
            except (AlphaVantageError, ValueError) as exc:
                logger.warning("Alpha Vantage refresh failed for %s: %s", symbol, exc)
                snapshot_id = _insert_fund_snapshot(
                    conn,
                    fund_symbol=symbol,
                    provider=ALPHA_VANTAGE_PROVIDER,
                    as_of_date=None,
                    status="failed",
                    holdings=[],
                    raw_payload={},
                    error_message=str(exc),
                )
                failed.append({"symbol": symbol, "snapshot_id": snapshot_id, "error": str(exc)})

        conn.commit()

    logger.info(
        "Completed investment fund holdings refresh candidates=%s refreshed=%s skipped=%s failed=%s",
        len(fund_symbols),
        len(refreshed),
        len(skipped),
        len(failed),
    )
    return {
        "status": "completed",
        "latest_import": latest_import,
        "fund_candidates": fund_symbols,
        "refreshed": refreshed,
        "skipped": skipped,
        "failed": failed,
    }


def _holdings_for_latest_fund_snapshots(conn, fund_symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for symbol in fund_symbols:
        snapshot = _latest_completed_fund_snapshot(conn, symbol)
        if not snapshot:
            logger.info("No completed fund holdings snapshot found for %s", symbol)
            continue
        rows = conn.execute(
            """
            SELECT constituent_symbol, constituent_name, weight_percent, asset_type, sector
            FROM investment_fund_holdings
            WHERE snapshot_id = ?
            ORDER BY weight_percent DESC, constituent_symbol ASC
            """,
            (snapshot["snapshot_id"],),
        ).fetchall()
        if rows:
            result[symbol] = [dict(row) for row in rows]
            result[symbol + "__snapshot"] = [snapshot]
            logger.info(
                "Loaded cached fund holdings for %s snapshot_id=%s holding_count=%s fetched_at=%s",
                symbol,
                snapshot["snapshot_id"],
                len(rows),
                snapshot["fetched_at"],
            )
        else:
            logger.info("Completed fund snapshot has no holdings for %s snapshot_id=%s", symbol, snapshot["snapshot_id"])
    return result


def get_single_name_exposure(
    *,
    min_percent: float = 1.0,
    limit: int = 15,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    logger.info("Building single-name exposure min_percent=%s limit=%s", min_percent, limit)
    with get_investments_connection(db_path) as conn:
        latest_import, positions = _latest_positions(conn)
        if latest_import is None:
            logger.info("Single-name exposure empty: no investment snapshot found")
            return {"status": "empty", "latest_import": None, "items": [], "summary": {}}

        total_value = sum(float(row.get("current_value") or 0) for row in positions)
        fund_symbols = _fund_candidates_from_positions(positions)
        logger.info(
            "Single-name exposure positions=%s total_value=%.2f fund_candidates=%s",
            len(positions),
            total_value,
            fund_symbols,
        )
        fund_holdings = _holdings_for_latest_fund_snapshots(conn, fund_symbols)

    exposures: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    excluded_value = 0.0
    resolved_value = 0.0

    for position in positions:
        symbol = position["symbol"]
        description = position.get("description") or ""
        position_value = float(position.get("current_value") or 0)
        if position_value <= 0:
            continue
        if position.get("is_cash_like") or _is_fixed_income_like(symbol, description):
            excluded_value += position_value
            continue

        cached_holdings = fund_holdings.get(symbol)
        if cached_holdings:
            snapshot = (fund_holdings.get(symbol + "__snapshot") or [{}])[0]
            allocated_for_position = 0.0
            for holding in cached_holdings:
                weight_percent = float(holding["weight_percent"] or 0)
                exposure_value = position_value * (weight_percent / 100)
                if exposure_value <= 0:
                    continue
                allocated_for_position += exposure_value
                item = exposures.setdefault(
                    holding["constituent_symbol"],
                    {
                        "symbol": holding["constituent_symbol"],
                        "name": holding["constituent_name"],
                        "exposure_value": 0.0,
                        "contributions": [],
                    },
                )
                item["exposure_value"] += exposure_value
                item["contributions"].append(
                    {
                        "source_symbol": symbol,
                        "source_description": description,
                        "source_value": _round_money(position_value) or 0.0,
                        "weight_percent": round(weight_percent, 4),
                        "exposure_value": _round_money(exposure_value) or 0.0,
                        "snapshot_as_of_date": snapshot.get("as_of_date"),
                        "snapshot_fetched_at": snapshot.get("fetched_at"),
                    }
                )
            resolved_value += allocated_for_position
        elif _is_fund_candidate(symbol, description, is_cash_like=bool(position.get("is_cash_like"))):
            unresolved.append(
                {
                    "symbol": symbol,
                    "description": description,
                    "value": _round_money(position_value) or 0.0,
                    "reason": "missing_fund_holdings_snapshot",
                }
            )
        else:
            item = exposures.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "name": description,
                    "exposure_value": 0.0,
                    "contributions": [],
                },
            )
            item["exposure_value"] += position_value
            item["contributions"].append(
                {
                    "source_symbol": symbol,
                    "source_description": description,
                    "source_value": _round_money(position_value) or 0.0,
                    "weight_percent": 100.0,
                    "exposure_value": _round_money(position_value) or 0.0,
                    "snapshot_as_of_date": latest_import.get("as_of_date"),
                    "snapshot_fetched_at": latest_import.get("imported_at"),
                }
            )
            resolved_value += position_value

    items = [
        {
            **item,
            "exposure_value": _round_money(item["exposure_value"]) or 0.0,
            "percent_of_portfolio": round((item["exposure_value"] / total_value) * 100, 2) if total_value else 0.0,
            "contributions": sorted(item["contributions"], key=lambda row: row["exposure_value"], reverse=True),
        }
        for item in exposures.values()
    ]
    items.sort(key=lambda item: item["exposure_value"], reverse=True)
    visible_items = [item for item in items if item["percent_of_portfolio"] >= min_percent][:limit]
    other_value = sum(item["exposure_value"] for item in items if item not in visible_items)

    logger.info(
        "Built single-name exposure total_items=%s visible_items=%s unresolved=%s excluded_value=%.2f resolved_value=%.2f",
        len(items),
        len(visible_items),
        len(unresolved),
        excluded_value,
        resolved_value,
    )

    return {
        "status": "ready",
        "latest_import": latest_import,
        "items": visible_items,
        "summary": {
            "total_value": _round_money(total_value) or 0.0,
            "resolved_exposure_value": _round_money(resolved_value) or 0.0,
            "resolved_exposure_percent": round((resolved_value / total_value) * 100, 2) if total_value else 0.0,
            "excluded_value": _round_money(excluded_value) or 0.0,
            "excluded_percent": round((excluded_value / total_value) * 100, 2) if total_value else 0.0,
            "unresolved_value": _round_money(sum(item["value"] for item in unresolved)) or 0.0,
            "unresolved_count": len(unresolved),
            "other_exposure_value": _round_money(other_value) or 0.0,
            "min_percent": min_percent,
            "limit": limit,
            "fund_candidate_count": len(fund_symbols),
        },
        "unresolved": unresolved,
    }


def _parse_json_object(raw: Any) -> dict[str, Any]:
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _fallback_industry_classification(item: dict[str, Any]) -> dict[str, Any]:
    symbol = _clean_text(item.get("symbol")).upper()
    name = _clean_text(item.get("name")).upper()
    haystack = f"{symbol} {name}"
    for needles, sector, industry in INDUSTRY_FALLBACK_RULES:
        if any(needle in haystack for needle in needles):
            return {
                "symbol": symbol,
                "name": name or symbol,
                "sector": sector,
                "industry": industry,
                "confidence": 0.72,
                "rationale": "Matched known public company pattern.",
                "provider": INDUSTRY_FALLBACK_PROVIDER,
            }
    return {
        "symbol": symbol,
        "name": name or symbol,
        "sector": "Unknown",
        "industry": "Unknown",
        "confidence": 0.35,
        "rationale": "No high-confidence local mapping.",
        "provider": INDUSTRY_FALLBACK_PROVIDER,
    }


def classify_investment_industries_with_llm(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("OPENAI_API_KEY missing; using deterministic industry classifier fallback")
        return [_fallback_industry_classification(item) for item in items]

    prompt_items = [
        {
            "symbol": item.get("symbol"),
            "name": item.get("name"),
            "percent_of_portfolio": item.get("percent_of_portfolio"),
        }
        for item in items
    ]
    llm = ChatOpenAI(
        model=os.getenv("FINANCE_INVESTMENT_CLASSIFIER_MODEL", os.getenv("FINANCE_CHAT_MODEL", "gpt-5.4-mini")),
        temperature=0,
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(content=INDUSTRY_CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps({"companies": prompt_items}, ensure_ascii=False, indent=2)),
            ]
        )
        parsed = _parse_json_object(response.content)
        classifications = parsed.get("classifications") or []
        by_symbol = {
            _clean_text(item.get("symbol")).upper(): item
            for item in classifications
            if _clean_text(item.get("symbol"))
        }
        normalized: list[dict[str, Any]] = []
        for item in items:
            symbol = _clean_text(item.get("symbol")).upper()
            raw = by_symbol.get(symbol) or {}
            fallback = _fallback_industry_classification(item)
            normalized.append(
                {
                    "symbol": symbol,
                    "name": _clean_text(raw.get("name")) or fallback["name"],
                    "sector": _clean_text(raw.get("sector")) or fallback["sector"],
                    "industry": _clean_text(raw.get("industry")) or fallback["industry"],
                    "confidence": max(0.0, min(float(raw.get("confidence") or fallback["confidence"]), 1.0)),
                    "rationale": _clean_text(raw.get("rationale")) or fallback["rationale"],
                    "provider": INDUSTRY_CLASSIFICATION_PROVIDER,
                }
            )
        return normalized
    except Exception as exc:
        logger.warning("Industry classifier LLM failed; using fallback classifications: %s", exc)
        return [_fallback_industry_classification(item) for item in items]


def _classification_cache_key(item: dict[str, Any]) -> tuple[str, str]:
    return _clean_text(item.get("symbol")).upper(), _clean_text(item.get("name")).upper()


def _load_industry_classification_cache(conn, items: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        symbol, name = _classification_cache_key(item)
        row = conn.execute(
            """
            SELECT symbol, name, sector, industry, confidence, rationale, provider, classified_at
            FROM investment_industry_classifications
            WHERE symbol = ? AND name = ?
            """,
            (symbol, name),
        ).fetchone()
        if row:
            cache[(symbol, name)] = dict(row)
    return cache


def _upsert_industry_classifications(conn, classifications: list[dict[str, Any]]) -> None:
    classified_at = _now_iso()
    conn.executemany(
        """
        INSERT INTO investment_industry_classifications (
            symbol, name, sector, industry, confidence, rationale, provider, classified_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, name)
        DO UPDATE SET
            sector = excluded.sector,
            industry = excluded.industry,
            confidence = excluded.confidence,
            rationale = excluded.rationale,
            provider = excluded.provider,
            classified_at = excluded.classified_at
        """,
        [
            (
                _clean_text(item.get("symbol")).upper(),
                _clean_text(item.get("name")).upper(),
                _clean_text(item.get("sector")) or "Unknown",
                _clean_text(item.get("industry")) or "Unknown",
                float(item.get("confidence") or 0),
                _clean_text(item.get("rationale")),
                _clean_text(item.get("provider")) or INDUSTRY_CLASSIFICATION_PROVIDER,
                classified_at,
            )
            for item in classifications
        ],
    )


def refresh_investment_industry_exposure(
    *,
    db_path: Optional[str] = None,
    force: bool = False,
    classifier: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] = classify_investment_industries_with_llm,
) -> dict[str, Any]:
    exposure = get_single_name_exposure(min_percent=0, limit=500, db_path=db_path)
    if exposure.get("status") != "ready":
        return {"status": exposure.get("status", "empty"), "snapshot_id": None, "items": [], "summary": {}}

    exposure_items = exposure.get("items") or []
    latest_import = exposure.get("latest_import") or {}
    total_value = float((exposure.get("summary") or {}).get("total_value") or 0)
    with get_investments_connection(db_path) as conn:
        cached = {} if force else _load_industry_classification_cache(conn, exposure_items)
        missing = [item for item in exposure_items if _classification_cache_key(item) not in cached]
        logger.info(
            "Refreshing industry exposure companies=%s cache_hits=%s missing=%s force=%s",
            len(exposure_items),
            len(cached),
            len(missing),
            force,
        )
        new_classifications = classifier(missing) if missing else []
        if new_classifications:
            _upsert_industry_classifications(conn, new_classifications)
        classifications = {
            **cached,
            **{(_clean_text(item.get("symbol")).upper(), _clean_text(item.get("name")).upper()): item for item in new_classifications},
        }

        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for item in exposure_items:
            key = _classification_cache_key(item)
            classification = classifications.get(key) or _fallback_industry_classification(item)
            sector = _clean_text(classification.get("sector")) or "Unknown"
            industry = _clean_text(classification.get("industry")) or "Unknown"
            group = groups.setdefault(
                (sector, industry),
                {
                    "sector": sector,
                    "industry": industry,
                    "exposure_value": 0.0,
                    "companies": [],
                },
            )
            exposure_value = float(item.get("exposure_value") or 0)
            group["exposure_value"] += exposure_value
            group["companies"].append(
                {
                    "symbol": item.get("symbol"),
                    "name": item.get("name"),
                    "exposure_value": _round_money(exposure_value) or 0.0,
                    "percent_of_portfolio": item.get("percent_of_portfolio") or 0.0,
                }
            )

        snapshot_id = str(uuid.uuid4())
        generated_at = _now_iso()
        grouped_items = sorted(groups.values(), key=lambda row: row["exposure_value"], reverse=True)
        conn.execute(
            """
            INSERT INTO investment_industry_exposure_snapshots (
                snapshot_id, source_import_id, generated_at, status, provider, raw_payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                latest_import.get("import_id"),
                generated_at,
                "completed",
                INDUSTRY_CLASSIFICATION_PROVIDER if os.getenv("OPENAI_API_KEY") else INDUSTRY_FALLBACK_PROVIDER,
                json.dumps({"classification_count": len(classifications)}, sort_keys=True),
            ),
        )
        conn.executemany(
            """
            INSERT INTO investment_industry_exposures (
                snapshot_id, sector, industry, exposure_value, percent_of_portfolio, company_count, top_companies_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot_id,
                    group["sector"],
                    group["industry"],
                    _round_money(group["exposure_value"]) or 0.0,
                    round((group["exposure_value"] / total_value) * 100, 2) if total_value else 0.0,
                    len(group["companies"]),
                    json.dumps(
                        sorted(group["companies"], key=lambda row: row["exposure_value"], reverse=True)[:6],
                        sort_keys=True,
                    ),
                )
                for group in grouped_items
            ],
        )
        conn.commit()

    result = get_investment_industry_exposure(db_path=db_path)
    return {
        **result,
        "classification_cache_hits": len(cached),
        "classification_requests": len(missing),
    }


def get_investment_industry_exposure(*, db_path: Optional[str] = None) -> dict[str, Any]:
    with get_investments_connection(db_path) as conn:
        snapshot = conn.execute(
            """
            SELECT snapshot_id, source_import_id, generated_at, status, provider
            FROM investment_industry_exposure_snapshots
            WHERE status = 'completed'
            ORDER BY generated_at DESC
            LIMIT 1
            """
        ).fetchone()
        if snapshot is None:
            return {
                "status": "empty",
                "snapshot": None,
                "items": [],
                "summary": {"total_exposure_value": 0.0, "industry_count": 0, "classified_at": None},
            }

        rows = conn.execute(
            """
            SELECT sector, industry, exposure_value, percent_of_portfolio, company_count, top_companies_json
            FROM investment_industry_exposures
            WHERE snapshot_id = ?
            ORDER BY exposure_value DESC
            """,
            (snapshot["snapshot_id"],),
        ).fetchall()

    items = []
    for row in rows:
        row_dict = dict(row)
        items.append(
            {
                "sector": row_dict["sector"],
                "industry": row_dict["industry"],
                "exposure_value": _round_money(row_dict["exposure_value"]) or 0.0,
                "percent_of_portfolio": row_dict["percent_of_portfolio"],
                "company_count": row_dict["company_count"],
                "top_companies": json.loads(row_dict.get("top_companies_json") or "[]"),
            }
        )
    return {
        "status": "ready",
        "snapshot": dict(snapshot),
        "items": items,
        "summary": {
            "total_exposure_value": _round_money(sum(item["exposure_value"] for item in items)) or 0.0,
            "industry_count": len(items),
            "classified_at": dict(snapshot).get("generated_at"),
            "provider": dict(snapshot).get("provider"),
        },
    }
