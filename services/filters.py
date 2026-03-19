def filter_ignored_payment(out):
    if "category_name" in out.columns:
        normalized = out["category_name"].fillna("").astype(str).str.strip().str.lower()
        payment_mask = normalized.isin({"ignored - income", "ignored - expense"})
        filtered_rows = out[payment_mask]
        print(f"[DEBUG] Filter candidate rows before exclusion: {filtered_rows.shape[0]}")
        out = out[~payment_mask]
    return out
