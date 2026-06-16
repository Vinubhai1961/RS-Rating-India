# =============================================================================
#   Filter high-RS, higher-priced stocks near 52-week highs
#
#   Enhanced version:
#   - Keeps original 52WKH/RS/price/volume filter behavior
#   - Adds optional output columns from enhanced RS pipeline:
#       ATR, ADR, SMA50, SMA200, SMA10W, SMA30W
#   - Adds safer parsing for K/M/B formatted volume and market-cap strings
#   - Adds distance-from-high, distance-from-low, and 52W range position
#   - Adds optional trend flags:
#       Above_SMA50, Above_SMA200, SMA50_Above_SMA200,
#       Above_SMA10W, Above_SMA30W, SMA10W_Above_SMA30W
#   - Uses robust column selection so missing columns do not break the script
# =============================================================================

import pandas as pd
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

# Optional trend filters.
# Keep False by default so this patch does not change your current selection logic.
REQUIRE_ABOVE_SMA50 = False
REQUIRE_ABOVE_SMA200 = False
REQUIRE_SMA50_ABOVE_SMA200 = False

REQUIRE_ABOVE_SMA10W = False
REQUIRE_ABOVE_SMA30W = False
REQUIRE_SMA10W_ABOVE_SMA30W = False
# ────────────────────────────────────────────────


def parse_number(x):
    """
    Parse numeric strings safely.

    Supports:
      498.95K
      11.00M
      2.45B
      300000
      1,234,567
      ₹1,234.56
      $1,234.56
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


def safe_bool(value):
    if pd.isna(value):
        return False
    return bool(value)


def add_derived_columns(df):
    """Add range, SMA, and volatility helper columns."""
    df["%_From_52WKH"] = ((df["Price"] - df["52WKH"]) / df["52WKH"]) * 100
    df["%_From_52WKH"] = df["%_From_52WKH"].round(2)

    df["%_Below_52WKH"] = (-df["%_From_52WKH"]).clip(lower=0).round(2)

    if "52WKL" in df.columns:
        df["%_Above_52WKL"] = ((df["Price"] - df["52WKL"]) / df["52WKL"]) * 100
        df["%_Above_52WKL"] = df["%_Above_52WKL"].round(2)

        range_width = df["52WKH"] - df["52WKL"]
        df["52W_Range_Pos"] = ((df["Price"] - df["52WKL"]) / range_width) * 100
        df.loc[range_width <= 0, "52W_Range_Pos"] = None
        df["52W_Range_Pos"] = df["52W_Range_Pos"].round(2)

    if "SMA50" in df.columns:
        df["Above_SMA50"] = df["Price"] > df["SMA50"]

    if "SMA200" in df.columns:
        df["Above_SMA200"] = df["Price"] > df["SMA200"]

    if "SMA50" in df.columns and "SMA200" in df.columns:
        df["SMA50_Above_SMA200"] = df["SMA50"] > df["SMA200"]

    if "SMA10W" in df.columns:
        df["Above_SMA10W"] = df["Price"] > df["SMA10W"]

    if "SMA30W" in df.columns:
        df["Above_SMA30W"] = df["Price"] > df["SMA30W"]

    if "SMA10W" in df.columns and "SMA30W" in df.columns:
        df["SMA10W_Above_SMA30W"] = df["SMA10W"] > df["SMA30W"]

    if "ATR" in df.columns:
        df["ATR_%"] = (df["ATR"] / df["Price"] * 100).round(2)

    if "ADR" in df.columns:
        df["ADR_%"] = (df["ADR"] / df["Price"] * 100).round(2)

    return df


def debug_ticker(df, ticker):
    row = df[df["Ticker"] == ticker]

    if row.empty:
        print(f"\nDEBUG: {ticker} → NOT FOUND in source data")
        return

    row = row.iloc[0]

    print(f"\n=== DEBUG: {ticker} ===")

    for label, col, fmt in [
        ("Price", "Price", ",.2f"),
        ("RS Percentile", "RS Percentile", ".1f"),
        ("10d Avg Vol", "AvgVol10", ",.0f"),
        ("52W High", "52WKH", ",.2f"),
        ("52W Low", "52WKL", ",.2f"),
        ("% from 52WH", "%_From_52WKH", ""),
        ("% below 52WH", "%_Below_52WKH", ""),
        ("% above 52WL", "%_Above_52WKL", ""),
        ("52W Range Pos", "52W_Range_Pos", ""),
        ("ATR", "ATR", ""),
        ("ADR", "ADR", ""),
        ("ATR_%", "ATR_%", ""),
        ("ADR_%", "ADR_%", ""),
        ("SMA50", "SMA50", ""),
        ("SMA200", "SMA200", ""),
        ("SMA10W", "SMA10W", ""),
        ("SMA30W", "SMA30W", ""),
    ]:
        if col in row.index:
            value = row.get(col)
            if pd.notna(value) and fmt:
                print(f"{label:<18}: {value:{fmt}}")
            else:
                print(f"{label:<18}: {value}")

    for col in [
        "Above_SMA50",
        "Above_SMA200",
        "SMA50_Above_SMA200",
        "Above_SMA10W",
        "Above_SMA30W",
        "SMA10W_Above_SMA30W",
    ]:
        if col in row.index:
            print(f"{col:<24}: {row.get(col)}")

    print("-" * 50)

    issues = []

    pct_from_high = row.get("%_From_52WKH")
    rs = row.get("RS Percentile")
    price = row.get("Price")
    vol = row.get("AvgVol10")

    if pd.isna(pct_from_high) or pct_from_high < -MAX_PCT_BELOW:
        issues.append(f"• Too far from 52WH ({pct_from_high}%)")

    if pd.isna(rs) or rs < RS_THRESHOLD:
        issues.append(f"• RS Percentile = {rs} (must be ≥ {RS_THRESHOLD})")

    if pd.isna(price) or price < PRICE_THRESHOLD:
        issues.append(f"• Price = {price} (must be ≥ {PRICE_THRESHOLD})")

    if pd.isna(vol) or vol < MIN_AVGVOL10:
        issues.append(f"• AvgVol10 = {vol} (must be ≥ {MIN_AVGVOL10:,})")

    optional_checks = [
        ("Above_SMA50", REQUIRE_ABOVE_SMA50),
        ("Above_SMA200", REQUIRE_ABOVE_SMA200),
        ("SMA50_Above_SMA200", REQUIRE_SMA50_ABOVE_SMA200),
        ("Above_SMA10W", REQUIRE_ABOVE_SMA10W),
        ("Above_SMA30W", REQUIRE_ABOVE_SMA30W),
        ("SMA10W_Above_SMA30W", REQUIRE_SMA10W_ABOVE_SMA30W),
    ]

    for col, enabled in optional_checks:
        if enabled and col in row.index and not safe_bool(row.get(col)):
            issues.append(f"• {col} failed")

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

    required_cols = ["Ticker", "Price", "52WKH", "RS Percentile", "AvgVol10"]
    missing_required = [c for c in required_cols if c not in df.columns]

    if missing_required:
        print(f"Error: Missing required columns: {missing_required}")
        return

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

    df = add_derived_columns(df)

    debug_ticker(df, DEBUG_TICKER)

    mask = (
        (df["%_From_52WKH"] >= -MAX_PCT_BELOW)
        & (df["RS Percentile"] >= RS_THRESHOLD)
        & (df["Price"] >= PRICE_THRESHOLD)
        & (df["AvgVol10"] >= MIN_AVGVOL10)
    )

    optional_filters = [
        ("Above_SMA50", REQUIRE_ABOVE_SMA50),
        ("Above_SMA200", REQUIRE_ABOVE_SMA200),
        ("SMA50_Above_SMA200", REQUIRE_SMA50_ABOVE_SMA200),
        ("Above_SMA10W", REQUIRE_ABOVE_SMA10W),
        ("Above_SMA30W", REQUIRE_ABOVE_SMA30W),
        ("SMA10W_Above_SMA30W", REQUIRE_SMA10W_ABOVE_SMA30W),
    ]

    for col, enabled in optional_filters:
        if enabled:
            if col in df.columns:
                mask = mask & df[col].fillna(False)
            else:
                print(f"Warning: optional filter {col} requested but column missing; ignoring.")

    filtered = df[mask].copy()

    print("\nAfter filters:")
    print(f"  • within {MAX_PCT_BELOW}% of 52-week high, including new highs")
    print(f"  • RS Percentile ≥ {RS_THRESHOLD}")
    print(f"  • Price ≥ {PRICE_THRESHOLD:,}")
    print(f"  • 10-day Avg Volume ≥ {MIN_AVGVOL10:,} shares")

    for col, enabled in optional_filters:
        if enabled:
            print(f"  • {col} = True")

    print(f"→ {len(filtered):,} rows remain")

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
        "ATR_%",
        "ADR",
        "ADR_%",
        "AvgVol",
        "AvgVol10",
        "52WKH",
        "52WKL",
        "%_From_52WKH",
        "%_Below_52WKH",
        "%_Above_52WKL",
        "52W_Range_Pos",
        "MCAP",
        "IPO",
        "SMA50",
        "SMA200",
        "SMA10W",
        "SMA30W",
        "Above_SMA50",
        "Above_SMA200",
        "SMA50_Above_SMA200",
        "Above_SMA10W",
        "Above_SMA30W",
        "SMA10W_Above_SMA30W",
    ]

    available = [c for c in desired if c in filtered.columns]

    sort_cols = [
        c
        for c in [
            "RS Percentile",
            "3M_RS Percentile",
            "1M_RS Percentile",
            "52W_Range_Pos",
        ]
        if c in filtered.columns
    ]

    ascending = [False for _ in sort_cols]

    if sort_cols:
        result = filtered[available].sort_values(
            sort_cols,
            ascending=ascending
        ).reset_index(drop=True)
    else:
        result = filtered[available].reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\nOutput overwritten → {OUTPUT_PATH}")
    print(f"Total rows saved: {len(result):,}")

    print("\nFirst 10 rows:")
    if len(result) > 0:
        print(result.head(10).to_string(index=False))
    else:
        print("No rows matched filters.")


if __name__ == "__main__":
    main()
