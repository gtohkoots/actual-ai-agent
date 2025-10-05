def filter_ignored_payment(out):
    if "category_name" in out.columns:
        payment_mask = (out["category_name"] == "Ignored - Income") | (out["category_name"] == "Ignored - expense")
        filtered_rows = out[payment_mask]
        print(f"[DEBUG] Filter candidate rows before exclusion: {filtered_rows.shape[0]}")
        out = out[~payment_mask]
    return out
