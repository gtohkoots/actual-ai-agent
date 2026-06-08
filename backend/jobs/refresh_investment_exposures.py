from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from backend.services.investments import refresh_investment_fund_holdings, refresh_investment_industry_exposure


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh cached ETF holdings for investment exposure charts.")
    parser.add_argument("--force", action="store_true", help="Refresh even when cached fund holdings are still fresh.")
    parser.add_argument("--max-age-hours", type=int, default=24, help="Cache age threshold before a fund is refreshed.")
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=1.2,
        help="Delay between Alpha Vantage requests to avoid free-tier burst limits.",
    )
    parser.add_argument("--skip-industry", action="store_true", help="Only refresh ETF holdings; do not rebuild industry exposure.")
    args = parser.parse_args()

    load_dotenv()
    result = {
        "fund_holdings": refresh_investment_fund_holdings(
            force=args.force,
            max_age_hours=args.max_age_hours,
            request_delay_seconds=args.request_delay_seconds,
        )
    }
    if not args.skip_industry:
        result["industry_exposure"] = refresh_investment_industry_exposure(force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
