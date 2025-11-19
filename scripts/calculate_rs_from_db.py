#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import arcticdb as adb
from tqdm.auto import tqdm

try:
    from pandas_market_calendars import get_calendar
except ImportError:
    get_calendar = None
    logging.warning("pandas_market_calendars not installed → RSRATING.csv will use consecutive days")

# === RS CORE FUNCTIONS (unchanged – perfect as-is) ===
def quarters_perf(closes: pd.Series, n: int) -> float:
    days = n * 63
    available_data = closes[-min(len(closes), days):]
    if len(available_data) < 1:
        return np.nan
    elif len(available_data) == 1:
        return 0.0
    pct_change = available_data.pct_change().dropna()
    return (pct_change + 1).cumprod().iloc[-1] - 1 if not pct_change.empty else np.nan

def strength(closes: pd.Series) -> float:
    perfs = [quarters_perf(closes, i) for i in range(1, 5)]
    valid_perfs = [p for p in perfs if not np.isnan(p)]
    if not valid_perfs:
        return np.nan
    weights = [0.4, 0.2, 0.2, 0.2][:len(valid_perfs)]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights] if total_weight > 0 else weights
    return sum(w * p for w, p in zip(weights, valid_perfs))

def relative_strength(closes: pd.Series, closes_ref: pd.Series) -> float:
    rs_stock = strength(closes)
    rs_ref = strength(closes_ref)
    if np.isnan(rs_stock) or np.isnan(rs_ref):
        return np.nan
    rs = (1 + rs_stock) / (1 + rs_ref) * 100
    return round(rs, 2) if rs <= 590 else np.nan

def load_arctic_db(data_dir):
    try:
        if not os.path.exists(data_dir):
            raise Exception(f"ArcticDB directory {data_dir} does not exist")
        arctic = adb.Arctic(f"lmdb://{data_dir}")
        if not arctic.has_library("prices"):
            raise Exception(f"No 'prices' library found in {data_dir}")
        lib = arctic.get_library("prices")
        symbols = lib.list_symbols()
        logging.info(f"Loaded {len(symbols):,} symbols from ArcticDB")
        return lib, symbols
    except Exception as e:
        logging.error(f"ArcticDB load failed: {str(e)}")
        print(f"ArcticDB error: {str(e)}")
        return None

# FIXED: Now uses NSE → XBOM → consecutive days
def generate_tradingview_csv(df_stocks, output_dir, ref_data, percentile_values=None):
    if percentile_values is None:
        percentile_values = [98, 89, 69, 49, 29, 9, 1]

    latest_ts = ref_data["datetime"].max()
    latest_date = datetime.fromtimestamp(latest_ts).date()
    logging.info(f"Latest market date: {latest_date} (India)")

    dates = []
    if get_calendar:
        for cal_name in ['NSE', 'XBOM']:
            try:
                cal = get_calendar(cal_name)
                sched = cal.schedule(start_date=latest_date - timedelta(days=20),
                                   end_date=latest_date + timedelta(days=2))
                valid_dates = [d.date() for d in sched.index if d.date() <= latest_date]
                dates = [d.strftime('%Y%m%dT') for d in valid_dates[-5:]]
                if len(dates) >= 5:
                    logging.info(f"Using {cal_name} calendar → {', '.join(dates)}")
                    break
            except Exception as e:
                logging.debug(f"{cal_name} calendar failed: {e}")
                continue

    if len(dates) < 5:
        dates = [(latest_date - timedelta(days=i)).strftime('%Y%m%dT') for i in range(4, -1, -1)]
        logging.info(f"Using consecutive dates → {', '.join(dates)}")

    # Percentile mapping
    rs_map = {}
    for p in percentile_values:
        rows = df_stocks[df_stocks["RS Percentile"] == p]
        rs_map[p] = rows.iloc[0]["RS"] if not rows.empty else 0.0

    lines = []
    for p in sorted(percentile_values, reverse=True):
        rs_val = rs_map[p]
        for d in dates:
            lines.append(f"{d},0,1000,0,{rs_val},0\n")

    with open(os.path.join(output_dir, "RSRATING.csv"), "w") as f:
        f.write(''.join(lines))

    logging.info(f"RSRATING.csv generated → {len(lines)} lines")
    print(f"RSRATING.csv ready for TradingView (India dates)")

def main(arctic_db_path, reference_ticker, output_dir, log_file, metadata_file=None, percentiles=None):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s | %(message)s")

    logging.info("=== INDIA RS CALCULATION STARTED ===")
    logging.info(f"Benchmark: {reference_ticker}")

    result = load_arctic_db(arctic_db_path)
    if not result:
        sys.exit(1)
    lib, tickers = result

    if reference_ticker not in tickers:
        logging.error(f"BENCHMARK {reference_ticker} NOT FOUND!")
        print(f"CRITICAL ERROR: {reference_ticker} missing from database!")
        sys.exit(1)

    ref_data = lib.read(reference_ticker).data
    ref_closes = pd.Series(ref_data["close"].values,
                           index=pd.to_datetime(ref_data["datetime"], unit='s'))
    logging.info(f"Benchmark {reference_ticker}: {len(ref_closes)} days of data")

    # Load metadata (your 67k list format)
    metadata_df = pd.DataFrame()
    if metadata_file and os.path.exists(metadata_file):
        try:
            with open(metadata_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                metadata = []
                for item in data:
                    info = item.get("info", {})
                    metadata.append({
                        "Ticker": item["ticker"],
                        "Price": info.get("Price"),
                        "DVol": info.get("DVol", ""),
                        "Sector": info.get("Sector", ""),
                        "Industry": info.get("Industry", ""),
                        "AvgVol": info.get("AvgVol", ""),
                        "AvgVol10": info.get("AvgVol10", ""),
                        "52WKH": info.get("52WKH"),
                        "52WKL": info.get("52WKL"),
                        "MCAP": info.get("MCAP"),
                        "Type": info.get("type", "Stock")
                    })
                metadata_df = pd.DataFrame(metadata)
                logging.info(f"Metadata loaded: {len(metadata_df):,} tickers")
        except Exception as e:
            logging.warning(f"Metadata failed: {e}")

    print(f"Calculating RS for {len(tickers)-1:,} stocks vs {reference_ticker}...")
    rs_results = []
    valid_count = 0

    for ticker in tqdm(tickers, desc="RS Calc"):
        if ticker == reference_ticker:
            continue
        try:
            data = lib.read(ticker).data
            closes = pd.Series(data["close"].values,
                               index=pd.to_datetime(data["datetime"], unit='s'))
            if len(closes) < 2:
                rs_results.append((ticker, np.nan, np.nan, np.nan, np.nan))
                continue

            rs = relative_strength(closes, ref_closes)
            rs_1m = relative_strength(closes[-20:], ref_closes[-20:]) if len(closes) >= 20 else np.nan
            rs_3m = relative_strength(closes[-60:], ref_closes[-60:]) if len(closes) >= 60 else np.nan
            rs_6m = relative_strength(closes[-120:], ref_closes[-120:]) if len(closes) >= 120 else np.nan

            rs_results.append((ticker, rs, rs_1m, rs_3m, rs_6m))
            if not np.isnan(rs):
                valid_count += 1
        except:
            rs_results.append((ticker, np.nan, np.nan, np.nan, np.nan))

    df = pd.DataFrame(rs_results, columns=["Ticker", "RS", "1M_RS", "3M_RS", "6M_RS"])
    if not metadata_df.empty:
        df = df.merge(metadata_df, on="Ticker", how="left")

    # Percentiles
    for col in ["RS", "1M_RS", "3M_RS", "6M_RS"]:
        valid = df[col].dropna()
        if not valid.empty:
            df[f"{col} Percentile"] = pd.qcut(valid.rank(method='first'), 100, labels=False, duplicates='drop')

    df = df.sort_values("RS", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    df["IPO"] = "No"
    if "Type" in df.columns:
        df.loc[df["Type"] == "ETF", ["Industry", "Sector"]] = "ETF"

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    df[["Rank", "Ticker", "Price", "DVol", "Sector", "Industry",
        "RS Percentile", "1M_RS Percentile", "3M_RS Percentile", "6M_RS Percentile",
        "AvgVol", "AvgVol10", "52WKH", "52WKL", "MCAP", "IPO"]].to_csv(
        os.path.join(output_dir, "rs_stocks.csv"), index=False, na_rep="")

    # Industries
    ind = df.groupby("Industry").agg({
        "RS Percentile": "mean", "1M_RS Percentile": "mean",
        "3M_RS Percentile": "mean", "6M_RS Percentile": "mean",
        "Sector": "first",
        "Ticker": lambda x: ",".join(sorted(x, key=lambda t: float(df[df["Ticker"]==t]["MCAP"].iloc[0] or 0), reverse=True))
    }).round(0).astype({"RS Percentile": int, "1M_RS Percentile": int,
                       "3M_RS Percentile": int, "6M_RS Percentile": int}).fillna(0)
    ind = ind.sort_values("RS Percentile", ascending=False).reset_index()
    ind["Rank"] = ind.index + 1
    ind.rename(columns={"RS Percentile": "RS", "1M_RS Percentile": "1 M_RS",
                        "3M_RS Percentile": "3M_RS", "6M_RS Percentile": "6M_RS"},
               inplace=True)
    ind[["Rank", "Industry", "Sector", "RS", "1 M_RS", "3M_RS", "6M_RS", "Ticker"]].to_csv(
        os.path.join(output_dir, "rs_industries.csv"), index=False)

    generate_tradingview_csv(df, output_dir, ref_data, percentiles)

    print(f"\nINDIA RS COMPLETE!")
    print(f"Valid RS: {valid_count:,} / {len(df):,}")
    print(f"Files → {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="India RS Rating vs NIFTYMIDSML400.NS")
    parser.add_argument("--arctic-db-path", default="tmp/arctic_db")
    parser.add_argument("--reference-ticker", default="NIFTYMIDSML400.NS")
    parser.add_argument("--output-dir", default="RS_Data")
    parser.add_argument("--log-file", default="RS_Logs/calc.log")
    parser.add_argument("--metadata-file", default="data/ticker_price.json")
    parser.add_argument("--percentiles", default="98,89,69,49,29,9,1")
    args = parser.parse_args()

    percentiles = [int(x) for x in args.percentiles.split(",")]
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    main(args.arctic_db_path, args.reference_ticker, args.output_dir,
         args.log_file, args.metadata_file, percentiles)
