from __future__ import annotations

from typing import Tuple

import pandas as pd

INTERNAL_TRANSFER_CATEGORIES = {
    "internal transfer income",
    "internal transfer expense",
}


def _normalize_category(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def is_internal_transfer_category(value: object) -> bool:
    return _normalize_category(value) in INTERNAL_TRANSFER_CATEGORIES


def is_internal_transfer_row(row: pd.Series) -> bool:
    if "category" in row:
        return is_internal_transfer_category(row.get("category"))
    if "category_name" in row:
        return is_internal_transfer_category(row.get("category_name"))
    return False


def filter_internal_transfer_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    if "category" in frame.columns:
        category_series = frame["category"]
    elif "category_name" in frame.columns:
        category_series = frame["category_name"]
    else:
        return frame.copy()

    mask = category_series.map(is_internal_transfer_category)
    return frame.loc[~mask].copy().reset_index(drop=True)


def split_internal_transfer_rows(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()

    if "category" in frame.columns:
        category_series = frame["category"]
    elif "category_name" in frame.columns:
        category_series = frame["category_name"]
    else:
        return frame.copy(), frame.iloc[0:0].copy()

    mask = category_series.map(is_internal_transfer_category)
    return frame.loc[~mask].copy().reset_index(drop=True), frame.loc[mask].copy().reset_index(drop=True)


def filter_ignored_payment(out):
    """Backward-compatible no-op retained for legacy callers."""
    return out
