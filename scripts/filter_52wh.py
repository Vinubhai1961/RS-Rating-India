# =============================================================================
#   Filter high-RS, higher-priced stocks near 52-week highs
# =============================================================================
import pandas as pd
from pathlib import Path

# ────────────────────────────────────────────────
#   CONFIG
# ────────────────────────────────────────────────
INPUT_PATH   = Path("RS_Data/rs_stocks.csv")
OUTPUT_PATH  = Path("RS_Data/RS80_Price30_within27pct_52wh.csv")

RS_THRESHOLD    = 77.0
PRICE_THRESHOLD = 30.0
MAX_PCT_BELOW   = 25.0
MIN_AVGVOL10    = 300_000

DEBUG_TICKER = "THERMAX.NS"
# ────────────────────────────────────────────────

def parse_volume(x):
    if pd.isna(x) or str(x).strip() in ['', 'nan', 'NaN']:
        return None
    x = str(x).strip().upper()
    if x.endswith('K'): return float(x[:-1]) * 1_000
    if x.endswith('M'): return float(x[:-1]) * 1_000_000
    if x.endswith('B'): return float(x[:-1]) * 1_000_000_000
    return float(x)


def debug_ticker(df, ticker):
    row = df[df['Ticker'] == ticker]
    if row.empty:
        print(f"\nDEBUG: {ticker} → NOT FOUND after cleaning")
        return
    
    row = row.iloc[0]
    price = row['Price']
    rs = row['RS Percentile']
    vol = row['AvgVol10']
    high = row['52WKH']
    
    print(f"\n=== DEBUG: {ticker} ===")
    print(f"Price          : ${price:,.2f}")
    print(f"RS Percentile  : {rs:.1f}")
    print(f"10d Avg Vol    : {'MISSING' if pd.isna(vol) else f'{vol:,.0f}'}")
    print(f"52W High       : {'MISSING' if pd.isna(high) else f'${high:,.2f}'}")
    print("-" * 40)

    if pd.isna(high):
        print("⚠️  52WKH is MISSING → Cannot calculate % from high. Including as potential opportunity.")
    else:
        pct_from_high = ((price - high) / high * 100).round(2)
        print(f"% from 52WH    : {pct_from_high}%")

    issues = []
    if pd.notna(high) and ((price - high) / high * 100 < -MAX_PCT_BELOW):
        issues.append(f"• Too far from 52WH ({((price - high)/high*100):.2f}%)")
    if rs < RS_THRESHOLD:
        issues.append(f"• RS Percentile = {rs} (must be ≥ {RS_THRESHOLD})")
    if price < PRICE_THRESHOLD:
        issues.append(f"• Price = ${price:,.2f} (must be ≥ ${PRICE_THRESHOLD})")
    
    if pd.notna(vol) and vol < MIN_AVGVOL10:
        issues.append(f"• AvgVol10 = {vol:,.0f} (must be ≥ {MIN_AVGVOL10:,})")
    elif pd.isna(vol):
        print("→ Volume MISSING → INCLUDED")

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

    # Convert numeric columns
    numeric_cols = ['Price', '52WKH', 'RS Percentile', 'AvgVol10']
    for col in numeric_cols:
        if col in df.columns:
            if col == 'AvgVol10':
                df[col] = df[col].apply(parse_volume)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Only drop rows missing truly critical data (Price + RS)
    df = df.dropna(subset=['Price', 'RS Percentile'])

    # Debug
    debug_ticker(df, DEBUG_TICKER)

    # Filter logic - allow missing 52WKH and missing volume
    mask = (
        (df['RS Percentile'] >= RS_THRESHOLD) &
        (df['Price'] >= PRICE_THRESHOLD) &
        ((df['AvgVol10'] >= MIN_AVGVOL10) | df['AvgVol10'].isna())
    )

    # Only apply 52WH filter if 52WKH exists
    if '52WKH' in df.columns:
        mask = mask & (
            df['52WKH'].isna() | 
            ((df['Price'] - df['52WKH']) / df['52WKH'] * 100 >= -MAX_PCT_BELOW)
        )

    filtered = df[mask].copy()

    print(f"\nAfter filters: {len(filtered):,} rows remain")

    # Add % from 52WH where possible
    if '52WKH' in filtered.columns:
        filtered['%_From_52WKH'] = ((filtered['Price'] - filtered['52WKH']) / filtered['52WKH'] * 100).round(2)

    # Save output
    desired = ['Rank', 'Ticker', 'Price', 'DVol', 'Sector', 'Industry', 'RS Percentile',
               '1M_RS Percentile', '3M_RS Percentile', '6M_RS Percentile',
               'AvgVol', 'AvgVol10', '52WKH', '52WKL', 'MCAP']

    available = [c for c in desired if c in filtered.columns]
    result = filtered[available].sort_values('RS Percentile', ascending=False).reset_index(drop=True)

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Output saved → {OUTPUT_PATH}")

    missing_vol = result['AvgVol10'].isna().sum()
    missing_high = result['52WKH'].isna().sum() if '52WKH' in result.columns else 0
    print(f"Missing Volume: {missing_vol} | Missing 52WH: {missing_high}")


if __name__ == "__main__":
    main()
