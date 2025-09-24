def filter_payment_amex(out, amex_id):
    if "payee" in out.columns and "account_pid" in out.columns:
        payment_mask = (out["payee"] == "Payment") & (out["account_pid"] == amex_id)
        filtered_rows = out[payment_mask]
        print(f"[DEBUG] Filter candidate rows before exclusion: {filtered_rows.shape[0]}")
        out = out[~payment_mask]
    return out
