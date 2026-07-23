# =============================================================================
#   RS70 + 60%+ Recovery from 52W Low + Stage 2 (Price > SMA10W > SMA30W)
#   WITH DETAILED DEBUG LOGGING
# =============================================================================
import pandas as pd
from pathlib import Path
from datetime import date

# ────────────────────────────────────────────────
#   CONFIG
# ────────────────────────────────────────────────
INPUT_PATH   = Path("RS_Data/rs_stocks.csv")
OUTPUT_PATH  = Path("RS_Data/RS70_Price30_52WKL.csv")
ARCHIVE_DIR  = Path("52wkl")
DEBUG_LOG    = Path("52wkl/debug_filter.log")

RS_THRESHOLD      = 70.0
PRICE_THRESHOLD   = 30.0
MIN_RECOVERY_PCT  = 60.0
MAX_PCT_TO_HIGH   = -27.0

MIN_AVGVOL10 = 300_000
MIN_ATR = 0
MIN_ADR = 0

# ────────────────────────────────────────────────

def parse_volume(x):
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    if x.endswith('K'): return float(x[:-1]) * 1_000
    if x.endswith('M'): return float(x[:-1]) * 1_000_000
    if x.endswith('B'): return float(x[:-1]) * 1_000_000_000
    return float(x)


def log_debug(ticker: str, message: str):
    """Append debug info to log file"""
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ticker}] {message}\n")


def main():
    if not INPUT_PATH.exists():
        print(f"Error: Input file not found → {INPUT_PATH}")
        return

    # Clear previous debug log
    if DEBUG_LOG.exists():
        DEBUG_LOG.unlink()

    print("Reading source file ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"→ Loaded {len(df):,} rows")

    # Convert numeric columns
    numeric_cols = ['Price', 'Prev_Close', '52WKH', '52WKL', 'RS Percentile', 
                    'AvgVol10', 'ATR', 'ADR', 'SMA10W', 'SMA30W', 'SMA20', 'SMA50', 'SMA200']
    for col in numeric_cols:
        if col in df.columns:
            if col == 'AvgVol10':
                df[col] = df[col].apply(parse_volume)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Price', '52WKH', '52WKL', 'RS Percentile', 'AvgVol10'])

    # Calculate metrics
    df['%_From_52WKL'] = ((df['Price'] - df['52WKL']) / df['52WKL']) * 100
    df['%_From_52WKL'] = df['%_From_52WKL'].round(2)
    
    df['%_From_52WKH'] = ((df['Price'] - df['52WKH']) / df['52WKH']) * 100
    df['%_From_52WKH'] = df['%_From_52WKH'].round(2)

    # Stage 2 Condition
    df['In_Stage2'] = (
        (df['Price'] > df['SMA10W']) & 
        (df['SMA10W'] > df['SMA30W'])
    )

    print("\n=== Starting Filter Debug ===")
    passed = []
    debug_count = 0

    for idx, row in df.iterrows():
        ticker = row['Ticker']
        debug_count += 1
        reasons = []

        # RS Check
        if row['RS Percentile'] < RS_THRESHOLD:
            reasons.append(f"RS {row['RS Percentile']:.0f} < {RS_THRESHOLD}")
        # Price Check
        if row['Price'] < PRICE_THRESHOLD:
            reasons.append(f"Price ${row['Price']:.2f} < ${PRICE_THRESHOLD}")
        # Recovery from 52W Low
        if row['%_From_52WKL'] < MIN_RECOVERY_PCT:
            reasons.append(f"Recovery {row['%_From_52WKL']}% < {MIN_RECOVERY_PCT}%")
        # Distance from 52W High
        if row['%_From_52WKH'] > MAX_PCT_TO_HIGH:
            reasons.append(f"Too close to 52WH: {row['%_From_52WKH']}%")
        # Volume
        if row['AvgVol10'] < MIN_AVGVOL10:
            reasons.append(f"AvgVol10 {row['AvgVol10']:,.0f} < {MIN_AVGVOL10:,}")
        # 52WKL > 1
        if row['52WKL'] <= 1:
            reasons.append(f"52WKL <= 1")
        # Stage 2
        if not row['In_Stage2']:
            reasons.append("Not in Stage 2 (Price > SMA10W > SMA30W)")

        # ATR/ADR for non-ETFs
        if row.get('Sector') != 'ETF' and pd.notna(row.get('Sector')):
            if row['ATR'] < MIN_ATR or row['ADR'] < MIN_ADR:
                reasons.append(f"ATR/ADR too low")

        if not reasons:
            passed.append(row)
            log_debug(ticker, "✅ PASSED ALL FILTERS")
        else:
            log_debug(ticker, "❌ FAILED: " + " | ".join(reasons))

        if debug_count % 500 == 0:
            print(f"Processed {debug_count:,} stocks...")

    filtered = pd.DataFrame(passed)

    print(f"\nAfter filters:")
    print(f"  • RS ≥ {RS_THRESHOLD}")
    print(f"  • Price ≥ ${PRICE_THRESHOLD}")
    print(f"  • Recovery from 52W Low ≥ {MIN_RECOVERY_PCT}%")
    print(f"  • More than 25% below 52W High")
    print(f"  • Stage 2: Price > SMA10W > SMA30W")
    print(f"→ {len(filtered):,} stocks remain")

    if len(filtered) == 0:
        print("No stocks match criteria. Check debug_filter.log for details.")
        return

    # Match original column structure
    desired = [
        'Rank', 'Ticker', 'Price', 'Prev_Close', 'DVol', 'Sector', 'Industry',
        'RS Percentile', '1M_RS Percentile', '3M_RS Percentile', '6M_RS Percentile',
        'ATR', 'ADR', 'AvgVol', 'AvgVol10', '52WKH', '52WKL', 'MCAP',
        'IPO', 'SMA20', 'SMA50', 'SMA200', 'SMA10W', 'SMA30W',
        '%_From_52WKL', '%_From_52WKH'
    ]

    available = [c for c in desired if c in filtered.columns]
    result = filtered[available].copy()

    result = result.sort_values(by='%_From_52WKL', ascending=False).reset_index(drop=True)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nOutput saved → {OUTPUT_PATH}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE_DIR / f"52wkl_{date.today().strftime('%m%d%Y')}.csv"
    result.to_csv(archive_path, index=False)
    print(f"Archive saved → {archive_path}")

    print(f"\nDetailed debug log → {DEBUG_LOG}")
    print("\nFirst 10 rows:")
    preview_cols = ['Rank', 'Ticker', 'Price', '%_From_52WKL', '%_From_52WKH', 
                   'RS Percentile', 'SMA10W', 'SMA30W', 'ATR', 'ADR']
    preview_cols = [c for c in preview_cols if c in result.columns]
    print(result.head(10)[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
