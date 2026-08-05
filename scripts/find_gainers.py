#!/usr/bin/env python3
"""
Find the day's top gainers from RS_Data/rs_stocks.csv.

A gainer is any ticker whose latest close is at least --min-gain percent above
the previous session's close:

    Gain_% = (Price - Prev_Close) / Prev_Close * 100

Both columns already exist in rs_stocks.csv, so this script does no market data
fetching at all -- it is a pure post-processing pass over the RS output.

Output:
    gainer/gainers_MMDDYYYY.csv

The output folder is created on the first run and reused after that.

Extra columns added on top of the rs_stocks.csv fields:
    Gain_%        percent move vs previous close
    Dist_52WH_%   how far below the 52-week high the ticker closed
    Above_SMA20 / Above_SMA50 / Above_SMA200   YES/NO trend context

Sort order: Gain_% descending, then Dist_52WH_% descending as the tiebreaker.

Usage:
    python scripts/find_gainers.py
    python scripts/find_gainers.py --min-gain 5 --input RS_Data/rs_stocks.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# Columns carried over from rs_stocks.csv, in output order. Anything missing
# from the input is silently skipped so this keeps working if the RS script's
# header changes again.
CARRY_COLUMNS = [
    "RS_Rank",
    "Ticker",
    "Price",
    "Prev_Close",
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
    "SMA20",
    "SMA50",
    "SMA200",
    "SMA10W",
    "SMA30W",
    "History_Days",
]

NEW_COLUMNS = [
    "Rank",
    "Gain_%",
    "Dist_52WH_%",
    "Above_SMA20",
    "Above_SMA50",
    "Above_SMA200",
]


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce a column to float, returning all-NaN if the column is absent."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def yes_no(series: pd.Series) -> pd.Series:
    """YES / NO / blank-on-missing, so the CSV never shows True/False/nan."""
    filled = series.fillna(False).astype(bool)
    return np.where(series.isna(), "", np.where(filled, "YES", "NO"))


def ensure_output_dir(path: str) -> str:
    """Create the output folder on first run; reuse it on later runs."""
    if os.path.isdir(path):
        logging.info("Output folder already exists: %s", path)
    else:
        os.makedirs(path, exist_ok=True)
        logging.info("Created output folder: %s", path)
    return path


def compute_gainers(df: pd.DataFrame, min_gain: float) -> pd.DataFrame:
    price = numeric(df, "Price")
    prev_close = numeric(df, "Prev_Close")

    # Guard against zero/negative prev_close, which would blow up the division
    # or produce a nonsense percentage.
    valid = price.notna() & prev_close.notna() & (prev_close > 0) & (price > 0)

    df = df.copy()
    df["Gain_%"] = np.where(valid, (price - prev_close) / prev_close * 100.0, np.nan)

    skipped = int((~valid).sum())
    if skipped:
        logging.info(
            "%s rows skipped: missing or invalid Price / Prev_Close", f"{skipped:,}"
        )

    gainers = df[df["Gain_%"] >= min_gain].copy()

    if gainers.empty:
        return gainers

    high_52w = numeric(gainers, "52WKH")
    close = numeric(gainers, "Price")
    gainers["Dist_52WH_%"] = np.where(
        (high_52w > 0) & close.notna(),
        ((high_52w - close) / high_52w * 100.0).clip(lower=0),
        np.nan,
    )

    # Trend context. NaN SMA (short history) stays blank rather than "NO", so a
    # 90-day IPO isn't misread as trading below its 200-day.
    for window in ["SMA20", "SMA50", "SMA200"]:
        sma = numeric(gainers, window)
        above = pd.Series(np.nan, index=gainers.index, dtype="object")
        comparable = sma.notna() & close.notna()
        above[comparable] = close[comparable] > sma[comparable]
        gainers[f"Above_{window}"] = yes_no(above.astype("boolean"))

    for col in ["Gain_%", "Dist_52WH_%"]:
        gainers[col] = pd.to_numeric(gainers[col], errors="coerce").round(2)

    # Both keys descending: biggest gain first, and where gains tie (common at
    # the 20% circuit limit) the ticker furthest below its 52-week high first.
    gainers = gainers.sort_values(
        ["Gain_%", "Dist_52WH_%"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    # Renumber Rank 1..N in gain order so the output reads as a standalone
    # leaderboard (1 = biggest gainer). The original universe-wide RS rank from
    # rs_stocks.csv is preserved as RS_Rank rather than being overwritten.
    if "Rank" in gainers.columns:
        gainers = gainers.rename(columns={"Rank": "RS_Rank"})
    gainers["Rank"] = gainers.index + 1

    return gainers


def order_columns(gainers: pd.DataFrame) -> list:
    """Gain fields first (that's the point of the file), then the RS context."""
    lead = ["Rank", "Ticker", "Price", "Prev_Close", "Gain_%", "Dist_52WH_%", "RS_Rank"]
    tail = [c for c in NEW_COLUMNS if c not in lead]
    carry = [c for c in CARRY_COLUMNS if c not in lead]
    ordered = lead + tail + carry
    return [c for c in ordered if c in gainers.columns]


def main():
    parser = argparse.ArgumentParser(
        description="Extract the day's top gainers (>= N%) from rs_stocks.csv"
    )
    parser.add_argument(
        "--input",
        default="RS_Data/rs_stocks.csv",
        help="Path to rs_stocks.csv (default: RS_Data/rs_stocks.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="gainer",
        help="Folder for gainer output (default: gainer)",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=5.0,
        help="Minimum percent gain to qualify (default: 5.0)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.0,
        help="Optional floor on Price to drop penny-stock noise (default: 0 = off)",
    )
    parser.add_argument(
        "--exclude-etf",
        action="store_true",
        help="Drop rows whose Sector is ETF",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not os.path.exists(args.input):
        logging.error("Input file not found: %s", args.input)
        sys.exit(1)

    df = pd.read_csv(args.input)
    logging.info("Loaded %s rows from %s", f"{len(df):,}", args.input)

    required = {"Price", "Prev_Close", "Ticker"}
    missing = required - set(df.columns)
    if missing:
        logging.error("Input is missing required column(s): %s", ", ".join(sorted(missing)))
        sys.exit(1)

    if args.exclude_etf and "Sector" in df.columns:
        before = len(df)
        df = df[~df["Sector"].astype(str).str.strip().str.upper().eq("ETF")]
        logging.info("Excluded %s ETF rows", f"{before - len(df):,}")

    if args.min_price > 0 and "Price" in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df["Price"], errors="coerce") >= args.min_price]
        logging.info(
            "Excluded %s rows below price floor %.2f", f"{before - len(df):,}", args.min_price
        )

    gainers = compute_gainers(df, args.min_gain)

    output_dir = ensure_output_dir(args.output_dir)
    stamp = datetime.now().strftime("%m%d%Y")
    out_path = os.path.join(output_dir, f"gainers_{stamp}.csv")

    if gainers.empty:
        logging.warning("No tickers gained %.2f%% or more today.", args.min_gain)
        # Still write a header-only file so downstream readers never 404.
        cols = list(dict.fromkeys(CARRY_COLUMNS + NEW_COLUMNS))
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
        print(f"\n=== GAINERS (>= {args.min_gain}%) ===")
        print("No qualifying tickers today.")
        print(f"Wrote: {out_path}")
        return

    cols = order_columns(gainers)
    gainers[cols].to_csv(out_path, index=False, na_rep="")

    print(f"\n=== GAINERS (>= {args.min_gain}%) ===")
    print(f"Universe scanned      : {len(df):,}")
    print(f"Qualifying gainers    : {len(gainers):,}")
    print(f"Best gain             : {gainers['Gain_%'].iloc[0]:.2f}%  ({gainers['Ticker'].iloc[0]})")
    print(f"Median gain           : {gainers['Gain_%'].median():.2f}%")

    print("\nTop 10:")
    show = [
        c
        for c in ["Rank", "Ticker", "Price", "Gain_%", "Dist_52WH_%", "RS Percentile", "Sector"]
        if c in gainers.columns
    ]
    print(gainers.head(10)[show].to_string(index=False))

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
