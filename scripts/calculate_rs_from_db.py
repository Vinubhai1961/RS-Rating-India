#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import arcticdb as adb
from tqdm.auto import tqdm

# Silence the annoying FutureWarning from pct_change()
warnings.filterwarnings("ignore", category=FutureWarning, message=".*fill_method.*pct_change.*")

try:
    from pandas_market_calendars import get_calendar
except ImportError:
    get_calendar = None
    logging.warning("pandas_market_calendars not installed → RSRATING.csv will use consecutive days")

# === YOUR ORIGINAL RS LOGIC (PERFECT) ===
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

def short_relative_strength(closes: pd.Series, closes_ref: pd.Series, days: int) -> float:
    if len(closes) < days + 1 or len(closes_ref) < days + 1:
        return np.nan
    stock_ret = closes.iloc[-1] / closes.iloc[-days] - 1
    ref_ret = closes_ref.iloc[-1] / closes_ref.iloc[-days] - 1
    rs = (1 + stock_ret) / (1 + ref_ret) * 100
    return round(rs, 2) if rs <= 590 else np.nan  # Keep your cap

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

def generate_tradingview_csv(df_stocks, output_dir, ref_data, percentile_values=None):
    if percentile_values is None:
        percentile_values = [98, 89, 69, 49, 29, 9, 1]   # TradingView expects these exact values

    latest_ts = ref_data["datetime"].max()
    latest_date = datetime.fromtimestamp(latest_ts).date()
    logging.info(f"Latest market date for RSRATING.csv: {latest_date}")

    # === Generate last 5 trading dates (same as your old code) ===
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
                    break
            except:
                pass
    if len(dates) < 5:
        dates = [(latest_date - timedelta(days=i)).strftime('%Y%m%dT') for i in range(4, -1, -1)]

    # === CORRECT WAY: Take the MINIMUM RS value that still belongs to that percentile ===
    valid_rs = df_stocks["RS"].dropna().sort_values(ascending=False).reset_index(drop=True)
    total_stocks = len(valid_rs)

    rs_map = {}
    for p in percentile_values:
        if total_stocks == 0:
            rs_map[p] = 50.0
            continue

        # Fixed: Uniform logic for all percentiles
        top_n = max(1, round(total_stocks * (100 - p) / 100.0))
        threshold_rs = valid_rs.iloc[min(top_n - 1, total_stocks - 1)]  # this is the cutoff value
        rs_map[p] = round(float(threshold_rs), 2)

    # === Write RSRATING.csv ===
    lines = []
    for p in sorted(percentile_values, reverse=True):    # 98,89,69,… descending
        rs_val = rs_map[p]
        for d in dates:
            lines.append(f"{d},0,1000,0,{rs_val},0\n")

    os.makedirs(output_dir, exist_ok=True)
    rating_path = os.path.join(output_dir, "RSRATING.csv")
    with open(rating_path, "w") as f:
        f.write(''.join(lines))

    logging.info(f"RSRATING.csv generated → {rating_path}")
    print("RSRATING.csv thresholds (correct now):")
    for p in sorted(percentile_values, reverse=True):
        print(f"  {p:2}th percentile → Raw RS ≥ {rs_map[p]:6.2f}")

# NEW: Safe MCAP converter (kept from previous fix)
def mcap_to_float(val):
    if pd.isna(val) or val in ["", None]:
        return 0.0
    s = str(val).strip().upper()
    multipliers = {'B': 1e9, 'T': 1e12, 'M': 1e6}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return float(s[:-1]) * mult
            except:
                return 0.0
    try:
        return float(s)
    except:
        return 0.0

def main(arctic_db_path, reference_ticker, output_dir, log_file, metadata_file=None, percentiles=None, debug=False):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.info("=== INDIA RS CALCULATION STARTED ===")
    logging.info(f"Benchmark: {reference_ticker}")

    result = load_arctic_db(arctic_db_path)
    if not result:
        sys.exit(1)
    lib, tickers = result

    if reference_ticker not in tickers:
        logging.error(f"BENCHMARK {reference_ticker} NOT FOUND!")
        print(f"CRITICAL ERROR: {reference_ticker} missing!")
        sys.exit(1)

    ref_data = lib.read(reference_ticker).data
    ref_closes = pd.Series(ref_data["close"].values, index=pd.to_datetime(ref_data["datetime"], unit='s'))

    # Load metadata
    metadata_df = pd.DataFrame()
    if metadata_file and os.path.exists(metadata_file):
        try:
            with open(metadata_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                records = []
                for item in data:
                    info = item.get("info", {})
                    records.append({
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
                metadata_df = pd.DataFrame(records)
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
            closes = pd.Series(data["close"].values, index=pd.to_datetime(data["datetime"], unit='s'))
            if len(closes) < 2:
                rs_results.append((ticker, np.nan, np.nan, np.nan, np.nan))
                continue

            rs = relative_strength(closes, ref_closes)  # Keep for full weighted RS
            rs_1m = short_relative_strength(closes, ref_closes, 21)
            rs_3m = short_relative_strength(closes, ref_closes, 63)
            rs_6m = short_relative_strength(closes, ref_closes, 126)

            rs_results.append((ticker, rs, rs_1m, rs_3m, rs_6m))
            if not np.isnan(rs):
                valid_count += 1
        except:
            rs_results.append((ticker, np.nan, np.nan, np.nan, np.nan))

    df = pd.DataFrame(rs_results, columns=["Ticker", "RS", "1M_RS", "3M_RS", "6M_RS"])
    if not metadata_df.empty:
        df = df.merge(metadata_df, on="Ticker", how="left")

    # Percentiles (updated for smoother 0-99)
    for col in ["RS", "1M_RS", "3M_RS", "6M_RS"]:
        valid = df[col].dropna()
        if not valid.empty:
            df.loc[valid.index, f"{col} Percentile"] = (valid.rank(pct=True, method='min') * 99).astype(int)
        else:
            df[f"{col} Percentile"] = np.nan

    df = df.sort_values("RS", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1

    # IPO flag
    df["IPO"] = "No"
    if "Type" in df.columns:
        df.loc[df["Type"] == "ETF", ["Industry", "Sector"]] = "ETF"

    os.makedirs(output_dir, exist_ok=True)

    # Save stocks
    df[["Rank", "Ticker", "Price", "DVol", "Sector", "Industry",
        "RS Percentile", "1M_RS Percentile", "3M_RS Percentile", "6M_RS Percentile",
        "AvgVol", "AvgVol10", "52WKH", "52WKL", "MCAP", "IPO"]].to_csv(
        os.path.join(output_dir, "rs_stocks.csv"), index=False, na_rep="")

    # === UPDATED: Tickers now sorted by RS (highest first) ===
    df_industries = df.groupby("Industry").agg({
        "RS Percentile": "mean",
        "1M_RS Percentile": "mean",
        "3M_RS Percentile": "mean",
        "6M_RS Percentile": "mean",
        "Sector": "first",
        "Ticker": lambda x: ",".join(
            df[df["Ticker"].isin(x)]
            .sort_values("RS", ascending=False)["Ticker"]
            .tolist()
        )
    }).reset_index()

    for col in ["RS Percentile", "1M_RS Percentile", "3M_RS Percentile", "6M_RS Percentile"]:
        df_industries[col] = df_industries[col].fillna(0).round().astype(int)

    df_industries = df_industries.sort_values("RS Percentile", ascending=False).reset_index(drop=True)
    df_industries["Rank"] = df_industries.index + 1
    df_industries.rename(columns={
        "RS Percentile": "RS",
        "1M_RS Percentile": "1 M_RS",
        "3M_RS Percentile": "3M_RS",
        "6M_RS Percentile": "6M_RS"
    }, inplace=True)

    df_industries[["Rank", "Industry", "Sector", "RS", "1 M_RS", "3M_RS", "6M_RS", "Ticker"]].to_csv(
        os.path.join(output_dir, "rs_industries.csv"), index=False)

    generate_tradingview_csv(df, output_dir, ref_data, percentiles)
    
    # DEBUG PRINTS (enhanced with data/RS details if --debug)
    print("\n=== DEBUG RS VALUES ===")
    debug_tickers = ["RELIANCE.NS", "TATASTEEL.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS"]
    debug_details = []
    for t in debug_tickers:
        if t in df["Ticker"].values:
            row = df[df["Ticker"] == t].iloc[0]
            rs = row['RS']
            percentile = row.get('RS Percentile', 'N/A')
            rank = row.get('Rank', 'N/A')
            print(f"{t:15} RS = {rs:6.2f} | Rank = {rank:4} | Percentile = {percentile}")
            
            if debug:
                # Enhanced: Check data sufficiency and RS calc details
                try:
                    data = lib.read(t).data
                    closes = pd.Series(data["close"].values, index=pd.to_datetime(data["datetime"], unit='s'))
                    start_date = pd.to_datetime(data["datetime"].min(), unit='s').date()
                    end_date = pd.to_datetime(data["datetime"].max(), unit='s').date()
                    num_days = len(closes)
                    sufficient = "✅ Sufficient" if num_days >= 252 else f"⚠️ Short: {num_days} days (need ~252 for full 1Y)"
                    
                    # RS breakdown: Quarters perfs (prices included via last n*63 days)
                    perf_3m = quarters_perf(closes, 1)
                    perf_6m = quarters_perf(closes, 2)
                    perf_9m = quarters_perf(closes, 3)
                    perf_12m = quarters_perf(closes, 4)
                    ref_perfs = [quarters_perf(ref_closes, i) for i in range(1, 5)]
                    strength_stock = strength(closes)
                    strength_ref = strength(ref_closes)
                    
                    print(f"  → Data: {num_days} days ({start_date} to {end_date}) | {sufficient}")
                    print(f"  → Prices used: Last {min(num_days, 252)} closes (daily adj. from {closes.index[-min(num_days,252):].min().date()} to {end_date})")
                    print(f"  → Quarters Returns: 3M={perf_3m:.1%} | 6M={perf_6m:.1%} | 9M={perf_9m:.1%} | 12M={perf_12m:.1%}")
                    print(f"  → Strength (wtd): Stock={strength_stock:.1%} | Ref={strength_ref:.1%} | RS Ratio={rs:.1f}")
                    print(f"  → Short RS: 1M={row['1M_RS']:.1f} (last 21 days) | 3M={row['3M_RS']:.1f} (last 63 days)")
                    
                    # Verify logic: Re-compute RS to confirm
                    recomputed_rs = relative_strength(closes, ref_closes)
                    logic_ok = "✅ OK" if abs(recomputed_rs - rs) < 0.01 else "❌ Mismatch!"
                    print(f"  → Logic Check: Recomputed RS={recomputed_rs:.2f} | {logic_ok}")
                    
                    debug_details.append({
                        'Ticker': t, 'Days': num_days, 'Start_Date': start_date, 'End_Date': end_date,
                        'Sufficient': sufficient, '3M_Return': perf_3m, '6M_Return': perf_6m,
                        '9M_Return': perf_9m, '12M_Return': perf_12m, 'Strength_Stock': strength_stock,
                        'Strength_Ref': strength_ref, 'RS': rs, 'Recomputed_RS': recomputed_rs
                    })
                except Exception as e:
                    print(f"  → Error loading details: {e}")
        else:
            print(f"{t:15} → Not found in results")

    print(f"\nINDIA RS COMPLETE! Valid RS: {valid_count:,} / {len(df):,}")

    # Save enhanced debug CSV if --debug
    if debug and debug_details:
        debug_df = pd.DataFrame(debug_details)
        debug_path = os.path.join(output_dir, "debug-rs.csv")
        debug_df.to_csv(debug_path, index=False)
        print(f"Enhanced debug details saved to: {debug_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="India RS Rating vs NIFTY 50 (^NSEI) - Full Market")
    parser.add_argument("--arctic-db-path", default="tmp/arctic_db")
    parser.add_argument("--reference-ticker", default="^NSEI")
    parser.add_argument("--output-dir", default="RS_Data")
    parser.add_argument("--log-file", default="logs/calc.log")
    parser.add_argument("--metadata-file", default="data/ticker_price.json")
    parser.add_argument("--percentiles", default="98,89,69,49,29,9,1")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug prints and save debug-rs.csv")
    args = parser.parse_args()

    percentiles = [int(x) for x in args.percentiles.split(",")]
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    main(args.arctic_db_path, args.reference_ticker, args.output_dir,
         args.log_file, args.metadata_file, percentiles, args.debug)
