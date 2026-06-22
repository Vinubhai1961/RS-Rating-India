# =============================================================================
#   Filter high-RS, higher-priced stocks near 52-week highs
#
#   Clean rebuild:
#   - Same filter logic as before
#   - Uses %_From_52WKH internally only
#   - Output columns match source rs_stocks.csv columns only
#   - Supports new columns already produced upstream:
#       ATR, ADR, SMA50, SMA200, SMA10W, SMA30W
# =============================================================================

import pandas as pd
from datetime import datetime
from pathlib import Path

# ────────────────────────────────────────────────
#   CONFIG
# ────────────────────────────────────────────────
INPUT_PATH = Path("RS_Data/rs_stocks.csv")
OUTPUT_PATH = Path("RS_Data/RS80_Price30_within25pct_52wh.csv")

RS_THRESHOLD = 77.0
PRICE_THRESHOLD = 30.0
MAX_PCT_BELOW = 25.0
MIN_AVGVOL10 = 300_000

DEBUG_TICKER = "HINDCOPPER.NS"
# ────────────────────────────────────────────────


def parse_number(x):
    """
    Parse K/M/B formatted values safely.

    Examples:
        498.95K -> 498950
        11.00M  -> 11000000
        2.45B   -> 2450000000
    """
    if pd.isna(x):
        return None

    if isinstance(x, (int, float)):
        return float(x)

    try:
        s = str(x).strip().upper()

        if s in {"", "NAN", "NONE", "NULL", "N/A"}:
            return None

        s = (
            s.replace(",", "")
             .replace("$", "")
             .replace("₹", "")
             .replace("RS.", "")
             .replace("INR", "")
             .strip()
        )

        multiplier = 1.0

        if s.endswith("K"):
            multiplier = 1_000
            s = s[:-1]
        elif s.endswith("M"):
            multiplier = 1_000_000
            s = s[:-1]
        elif s.endswith("B"):
            multiplier = 1_000_000_000
            s = s[:-1]
        elif s.endswith("T"):
            multiplier = 1_000_000_000_000
            s = s[:-1]

        return float(s) * multiplier

    except Exception:
        return None


def debug_ticker(df, ticker):
    row = df[df["Ticker"] == ticker]

    if row.empty:
        print(f"\nDEBUG: {ticker} → NOT FOUND in source data")
        return

    row = row.iloc[0]

    price = row.get("Price")
    rs = row.get("RS Percentile")
    vol = row.get("AvgVol10")
    high = row.get("52WKH")
    pct_from_high = row.get("%_From_52WKH")

    print(f"\n=== DEBUG: {ticker} ===")
    print(f"Price          : {price:,.2f}" if pd.notna(price) else "Price          : n/a")
    print(f"RS Percentile  : {rs:.1f}" if pd.notna(rs) else "RS Percentile  : n/a")
    print(f"10d Avg Vol    : {vol:,.0f}" if pd.notna(vol) else "10d Avg Vol    : n/a")
    print(f"52W High       : {high:,.2f}" if pd.notna(high) else "52W High       : n/a")
    print(f"% from 52WH    : {pct_from_high}%")

    for col in ["ATR", "ADR", "SMA50", "SMA200", "SMA10W", "SMA30W"]:
        if col in row.index:
            print(f"{col:<15}: {row.get(col)}")

    print("-" * 40)

    issues = []

    if pd.isna(pct_from_high) or pct_from_high < -MAX_PCT_BELOW:
        issues.append(f"• Too far from 52WH ({pct_from_high}%)")

    if pd.isna(rs) or rs < RS_THRESHOLD:
        issues.append(f"• RS Percentile = {rs} (must be ≥ {RS_THRESHOLD})")

    if pd.isna(price) or price < PRICE_THRESHOLD:
        issues.append(f"• Price = {price} (must be ≥ {PRICE_THRESHOLD})")

    if pd.isna(vol) or vol < MIN_AVGVOL10:
        issues.append(f"• AvgVol10 = {vol} (must be ≥ {MIN_AVGVOL10:,})")

    if not issues:
        print("→ PASSED ALL FILTERS ✓")
    else:
        print("→ FAILED FILTERS:")
        for issue in issues:
            print(issue)


def main():
    if not INPUT_PATH.exists():
        print(f"Error: Input file not found → {INPUT_PATH}")
        return

    print("Reading source file ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"→ Loaded {len(df):,} rows")
    print(f"→ Columns found: {list(df.columns)}")

    required_cols = [
        "Ticker",
        "Price",
        "52WKH",
        "RS Percentile",
        "AvgVol10",
    ]

    missing_required = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_required:
        print(f"Error: Missing required columns: {missing_required}")
        return

    # Convert required numeric/filter columns.
    numeric_cols = [
        "Price",
        "52WKH",
        "52WKL",
        "RS Percentile",
        "1M_RS Percentile",
        "3M_RS Percentile",
        "6M_RS Percentile",
        "AvgVol",
        "AvgVol10",
        "DVol",
        "MCAP",
        "ATR",
        "ADR",
        "SMA50",
        "SMA200",
        "SMA10W",
        "SMA30W",
    ]

    for col in numeric_cols:
        if col in df.columns:
            if col in {"AvgVol", "AvgVol10", "DVol", "MCAP"}:
                df[col] = df[col].apply(parse_number)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    before_drop = len(df)

    df = df.dropna(
        subset=[
            "Price",
            "52WKH",
            "RS Percentile",
            "AvgVol10",
        ]
    )

    dropped = before_drop - len(df)
    print(f"→ Dropped {dropped:,} rows missing required numeric fields")

    # Internal-only helper column for filtering.
    # Negative value means price is below 52WKH.
    # 0 means exactly at 52WKH.
    # Positive value means new high / above stored 52WKH.
    df["%_From_52WKH"] = ((df["Price"] - df["52WKH"]) / df["52WKH"]) * 100
    df["%_From_52WKH"] = df["%_From_52WKH"].round(2)

    debug_ticker(df, DEBUG_TICKER)

    # Same filter logic:
    # allow new highs and allow stocks up to MAX_PCT_BELOW below 52WKH.
    mask = (
        (df["%_From_52WKH"] >= -MAX_PCT_BELOW)
        & (df["RS Percentile"] >= RS_THRESHOLD)
        & (df["Price"] >= PRICE_THRESHOLD)
        & (df["AvgVol10"] >= MIN_AVGVOL10)
    )

    filtered = df[mask].copy()

    print("\nAfter filters:")
    print(f"  • within {MAX_PCT_BELOW}% of 52-week high, including new highs")
    print(f"  • RS Percentile ≥ {RS_THRESHOLD}")
    print(f"  • Price ≥ {PRICE_THRESHOLD:,}")
    print(f"  • 10-day Avg Volume ≥ {MIN_AVGVOL10:,} shares")
    print(f"→ {len(filtered):,} rows remain")

    # Output exactly same style/source columns only.
    desired = [
        "Rank",
        "Ticker",
        "Price",
        "DVol",
        "Sector",
        "Industry",
        "RS Percentile",
        "1M_RS Percentile",
        "3M_RS Percentile",
        "6M_RS Percentile",
        "ATR",
        "ADR",
        "AvgVol",
        "AvgVol10",
        "52WKH",
        "52WKL",
        "MCAP",
        "IPO",
        "SMA50",
        "SMA200",
        "SMA10W",
        "SMA30W",
    ]

    available = [
        col for col in desired
        if col in filtered.columns
    ]

    # Keep sorting simple and close to original behavior.
    sort_cols = [
        col for col in [
            "RS Percentile",
            "3M_RS Percentile",
            "1M_RS Percentile",
        ]
        if col in filtered.columns
    ]

    if sort_cols:
        result = (
            filtered[available]
            .sort_values(sort_cols, ascending=[False] * len(sort_cols))
            .reset_index(drop=True)
        )
    else:
        result = filtered[available].reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Daily output (existing)
    result.to_csv(OUTPUT_PATH, index=False)
    
    # Archive output
    archive_dir = Path("52wh")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    current_date = datetime.now().strftime("%Y%m%d")
    archive_file = archive_dir / f"52wh_{current_date}.csv"
    
    result.to_csv(archive_file, index=False)
    
    print(f"\nOutput overwritten → {OUTPUT_PATH}")
    print(f"Archive saved     → {archive_file}")
    print(f"Total rows saved: {len(result):,}")


if __name__ == "__main__":
    main()
