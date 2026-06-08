import sqlite3
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import app
from backend.services.investments import (
    get_investment_industry_exposure,
    get_investments_overview,
    get_single_name_exposure,
    import_fidelity_positions_csv,
    refresh_investment_industry_exposure,
    refresh_investment_fund_holdings,
)
from backend.services.market_data.alpha_vantage import parse_etf_profile_payload
from backend.utils.investments_db import init_investments_db


FIDELITY_CSV = """Account Number,Account Name,Symbol,Description,Quantity,Last Price,Last Price Change,Current Value,Today's Gain/Loss Dollar,Today's Gain/Loss Percent,Total Gain/Loss Dollar,Total Gain/Loss Percent,Percent Of Account,Cost Basis Total,Average Cost Basis,Type
Z00000001,Individual,SPAXX**,HELD IN MONEY MARKET,,,,$3.31,,,,,0.14%,,,Cash,
Z00000001,Individual,QQQ,INVESCO QQQ TR UNIT SER 1,1.091,$738.31,+$2.71,$805.49,+$2.95,+0.36%,+$6.75,+0.84%,33.30%,$798.74,$732.12,Cash,
Downloaded from Fidelity
"""

FIDELITY_TWO_FUNDS_CSV = """Account Number,Account Name,Symbol,Description,Quantity,Last Price,Last Price Change,Current Value,Today's Gain/Loss Dollar,Today's Gain/Loss Percent,Total Gain/Loss Dollar,Total Gain/Loss Percent,Percent Of Account,Cost Basis Total,Average Cost Basis,Type
Z00000001,Individual,QQQ,INVESCO QQQ TR UNIT SER 1,1,$100,,$100,,,,,50%,$100,$100,Cash,
Z00000001,Individual,VOO,VANGUARD INDEX FUNDS S&P 500 ETF USD,1,$100,,$100,,,,,50%,$100,$100,Cash,
"""


def test_init_investments_db_creates_tables(tmp_path):
    db_path = tmp_path / "investments.sqlite"

    init_investments_db(str(db_path))

    with sqlite3.connect(db_path) as conn:
      rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()

    table_names = {row[0] for row in rows}
    assert {
        "investment_imports",
        "investment_accounts",
        "investment_positions",
        "investment_fund_snapshots",
        "investment_fund_holdings",
        "investment_industry_classifications",
        "investment_industry_exposure_snapshots",
        "investment_industry_exposures",
    } <= table_names


def test_import_fidelity_positions_csv_and_overview(tmp_path):
    db_path = tmp_path / "investments.sqlite"

    result = import_fidelity_positions_csv(
        FIDELITY_CSV.encode("utf-8-sig"),
        filename="positions.csv",
        as_of_date="2026/05/30",
        db_path=str(db_path),
    )
    overview = get_investments_overview(db_path=str(db_path))

    assert result["row_count"] == 2
    assert result["account_count"] == 1
    assert result["total_value"] == 808.8
    assert result["as_of_date"] == "2026-05-30"
    assert overview["status"] == "ready"
    assert overview["summary"]["total_value"] == 808.8
    assert overview["summary"]["cash_value"] == 3.31
    assert overview["summary"]["holding_count"] == 2
    assert overview["summary"]["total_gain_loss_amount"] == 6.75
    assert overview["accounts"][0]["account_name"] == "Individual"
    assert overview["holdings"][0]["symbol"] == "QQQ"
    assert overview["holdings"][0]["percent_of_portfolio"] == 99.59
    assert overview["holdings"][1]["is_cash_like"] is True


def test_investments_api_import_and_overview(monkeypatch, tmp_path):
    db_path = tmp_path / "investments.sqlite"
    monkeypatch.setenv("FINANCE_INVESTMENTS_DB_PATH", str(db_path))

    client = TestClient(app)
    import_response = client.post(
        "/api/investments/import/fidelity-positions-csv",
        params={"filename": "positions.csv", "as_of_date": "2026-05-30"},
        content=FIDELITY_CSV.encode("utf-8-sig"),
        headers={"content-type": "text/csv"},
    )
    overview_response = client.get("/api/investments/overview")

    assert import_response.status_code == 200
    assert import_response.json()["row_count"] == 2
    assert overview_response.status_code == 200
    assert overview_response.json()["summary"]["total_value"] == 808.8


def test_investments_api_rejects_unknown_csv(monkeypatch, tmp_path):
    db_path = tmp_path / "investments.sqlite"
    monkeypatch.setenv("FINANCE_INVESTMENTS_DB_PATH", str(db_path))

    client = TestClient(app)
    response = client.post(
        "/api/investments/import/fidelity-positions-csv",
        content=b"not,a,fidelity,file\n1,2,3\n",
        headers={"content-type": "text/csv"},
    )

    assert response.status_code == 400


def test_parse_alpha_vantage_etf_profile_payload():
    payload = {
        "net_assets": "1000000",
        "latest_holding_date": "2026-05-31",
        "holdings": [
            {"symbol": "AAPL", "description": "Apple Inc", "weight": "8.5%", "sector": "Technology"},
            {"symbol": "MSFT", "description": "Microsoft Corp", "weight": "0.071"},
            {"symbol": "", "description": "Footer", "weight": ""},
        ],
    }

    parsed = parse_etf_profile_payload("QQQ", payload)

    assert parsed["fund_symbol"] == "QQQ"
    assert parsed["as_of_date"] == "2026-05-31"
    assert parsed["holdings"] == [
        {
            "symbol": "AAPL",
            "name": "APPLE INC",
            "weight_percent": 8.5,
            "asset_type": "",
            "sector": "Technology",
        },
        {
            "symbol": "MSFT",
            "name": "MICROSOFT CORP",
            "weight_percent": 7.1,
            "asset_type": "",
            "sector": "",
        },
    ]


def test_parse_alpha_vantage_niche_etf_payload_normalizes_missing_symbols():
    payload = {
        "holdings": [
            {"symbol": "n/a", "description": "SK HYNIX INC", "weight": "0.2598"},
            {"symbol": "n/a", "description": "SK HYNIX INC-SWAP-GOLD-L", "weight": "0.0106"},
            {"symbol": "n/a", "description": "SAMSUNG ELECTRONICS CO LTD", "weight": "0.184"},
            {"symbol": "n/a", "description": "SAMSUNG ELECTRONICS -SWAP-GOLD-L", "weight": "0.0075"},
            {"symbol": "n/a", "description": "MICRON TECHNOLOGY INC SWAP NM", "weight": "0.1332"},
            {"symbol": "FGXXX", "description": "FIRST AMERICAN GOVERNMENT OBLIGS X", "weight": "0.1194"},
            {"symbol": "MU", "description": "MICRON TECHNOLOGY INC", "weight": "0.0532"},
            {"symbol": "n/a", "description": "OTHER ASSETS AND LIABILITIES", "weight": "0.0253"},
            {"symbol": "n/a", "description": "US DOLLARS", "weight": "-0.0125"},
            {"symbol": "n/a", "description": "CASH OFFSET", "weight": "-0.1774"},
        ],
    }

    parsed = parse_etf_profile_payload("DRAM", payload)

    assert parsed["holdings"] == [
        {
            "symbol": "SK HYNIX INC",
            "name": "SK HYNIX INC",
            "weight_percent": 27.04,
            "asset_type": "",
            "sector": "",
        },
        {
            "symbol": "SAMSUNG ELECTRONICS CO LTD",
            "name": "SAMSUNG ELECTRONICS CO LTD",
            "weight_percent": 19.15,
            "asset_type": "",
            "sector": "",
        },
        {
            "symbol": "MU",
            "name": "MICRON TECHNOLOGY INC",
            "weight_percent": 18.64,
            "asset_type": "",
            "sector": "",
        },
    ]


def test_refresh_fund_holdings_and_single_name_exposure(tmp_path):
    db_path = tmp_path / "investments.sqlite"
    import_fidelity_positions_csv(
        FIDELITY_CSV,
        filename="positions.csv",
        as_of_date="2026-05-30",
        db_path=str(db_path),
    )

    def fake_fetcher(symbol):
        assert symbol == "QQQ"
        return {
            "fund_symbol": symbol,
            "as_of_date": "2026-05-31",
            "holdings": [
                {"symbol": "AAPL", "name": "Apple Inc", "weight_percent": 50.0},
                {"symbol": "MSFT", "name": "Microsoft Corp", "weight_percent": 25.0},
            ],
            "raw_payload": {"symbol": symbol},
        }

    refresh_result = refresh_investment_fund_holdings(db_path=str(db_path), profile_fetcher=fake_fetcher)
    exposure = get_single_name_exposure(db_path=str(db_path), min_percent=1.0, limit=10)

    assert refresh_result["fund_candidates"] == ["QQQ"]
    assert refresh_result["refreshed"][0]["holding_count"] == 2
    assert exposure["status"] == "ready"
    assert exposure["items"][0]["symbol"] == "AAPL"
    assert exposure["items"][0]["exposure_value"] == 402.75
    assert exposure["items"][0]["percent_of_portfolio"] == 49.8
    assert exposure["summary"]["excluded_value"] == 3.31
    assert exposure["summary"]["unresolved_count"] == 0


def test_refresh_fund_holdings_throttles_alpha_vantage_requests(tmp_path):
    db_path = tmp_path / "investments.sqlite"
    import_fidelity_positions_csv(FIDELITY_TWO_FUNDS_CSV, db_path=str(db_path))
    fetched_symbols = []
    slept_for = []

    def fake_fetcher(symbol):
        fetched_symbols.append(symbol)
        return {
            "fund_symbol": symbol,
            "as_of_date": "2026-05-31",
            "holdings": [{"symbol": "AAPL", "name": "Apple Inc", "weight_percent": 100.0}],
            "raw_payload": {"symbol": symbol},
        }

    refresh_result = refresh_investment_fund_holdings(
        db_path=str(db_path),
        request_delay_seconds=1.2,
        profile_fetcher=fake_fetcher,
        sleeper=slept_for.append,
    )

    assert fetched_symbols == ["QQQ", "VOO"]
    assert slept_for == [1.2]
    assert len(refresh_result["refreshed"]) == 2


def test_refresh_industry_exposure_groups_classified_companies(tmp_path):
    db_path = tmp_path / "investments.sqlite"
    import_fidelity_positions_csv(FIDELITY_CSV, db_path=str(db_path))

    def fake_fetcher(symbol):
        return {
            "fund_symbol": symbol,
            "as_of_date": "2026-05-31",
            "holdings": [
                {"symbol": "AAPL", "name": "APPLE INC", "weight_percent": 50.0},
                {"symbol": "MSFT", "name": "MICROSOFT CORP", "weight_percent": 25.0},
            ],
            "raw_payload": {"symbol": symbol},
        }

    def fake_classifier(items):
        return [
            {
                "symbol": item["symbol"],
                "name": item["name"],
                "sector": "Information Technology",
                "industry": "Consumer Electronics" if item["symbol"] == "AAPL" else "Software",
                "confidence": 0.99,
                "rationale": "test fixture",
                "provider": "test",
            }
            for item in items
        ]

    refresh_investment_fund_holdings(db_path=str(db_path), profile_fetcher=fake_fetcher)
    result = refresh_investment_industry_exposure(db_path=str(db_path), classifier=fake_classifier)
    cached = get_investment_industry_exposure(db_path=str(db_path))

    assert result["status"] == "ready"
    assert result["classification_requests"] == 2
    assert result["summary"]["industry_count"] == 2
    assert cached["items"][0]["industry"] == "Consumer Electronics"
    assert cached["items"][0]["percent_of_portfolio"] == 49.8


def test_investment_industry_exposure_api(monkeypatch, tmp_path):
    db_path = tmp_path / "investments.sqlite"
    monkeypatch.setenv("FINANCE_INVESTMENTS_DB_PATH", str(db_path))
    import_fidelity_positions_csv(FIDELITY_CSV, db_path=str(db_path))

    def fake_fetcher(symbol):
        return {
            "fund_symbol": symbol,
            "as_of_date": "2026-05-31",
            "holdings": [{"symbol": "AAPL", "name": "APPLE INC", "weight_percent": 100.0}],
            "raw_payload": {"symbol": symbol},
        }

    refresh_investment_fund_holdings(db_path=str(db_path), profile_fetcher=fake_fetcher)
    refresh_investment_industry_exposure(db_path=str(db_path))
    client = TestClient(app)

    response = client.get("/api/investments/industry/exposure")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["items"][0]["industry"] == "Consumer Electronics"


def test_investment_exposure_api(monkeypatch, tmp_path):
    db_path = tmp_path / "investments.sqlite"
    monkeypatch.setenv("FINANCE_INVESTMENTS_DB_PATH", str(db_path))
    import_fidelity_positions_csv(FIDELITY_CSV, db_path=str(db_path))

    def fake_fetcher(symbol):
        return {
            "fund_symbol": symbol,
            "as_of_date": "2026-05-31",
            "holdings": [{"symbol": "AAPL", "name": "Apple Inc", "weight_percent": 100.0}],
            "raw_payload": {"symbol": symbol},
        }

    refresh_investment_fund_holdings(db_path=str(db_path), profile_fetcher=fake_fetcher)
    client = TestClient(app)

    response = client.get("/api/investments/exposure/single-name", params={"min_percent": 1, "limit": 5})

    assert response.status_code == 200
    assert response.json()["items"][0]["symbol"] == "AAPL"
