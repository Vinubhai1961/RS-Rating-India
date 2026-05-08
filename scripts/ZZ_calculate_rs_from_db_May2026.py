#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
print(f"Using pandas {pd.__version__}")
import numpy as np
import arcticdb as adb
from tqdm.auto import tqdm

try:
    from pandas_market_calendars import get_calendar
except ImportError:
    get_calendar = None
    logging.warning("pandas_market_calendars not installed. Falling back to consecutive days for RSRATING.csv.")


# ====================== YOUR NEW FUNCTION ======================
def calculate_smas_and_adr(df):
    """Robust calculation of SMA50, SMA200 (daily), SMA10/30 Weekly, and ADR"""
    sma50 = sma200 = sma10w = sma30w = adr = np.nan
    if 'close' not in df.columns:
        return sma50, sma200, sma10w, sma30w, adr
    
    closes = pd.to_numeric(df['close'], errors='coerce').dropna()
    
    # Daily SMA
    if len(closes) >= 50:
        sma50 = round(closes.rolling(50).mean().iloc[-1], 2)
    if len(closes) >= 200:
        sma200 = round(closes.rolling(200).mean().iloc[-1], 2)
    
    # Weekly SMA
    if len(closes) >= 30:
        weekly_closes = closes.resample('W-FRI').last().dropna()
        if len(weekly_closes) >= 10:
            sma10w = round(weekly_closes.rolling(10).mean().iloc[-1], 2)
        if len(weekly_closes) >= 30:
            sma30w = round(weekly_closes.rolling(30).mean().iloc[-1], 2)
    
    # ADR
    if all(col in df.columns for col in ['high', 'low']):
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        daily_range_pct = ((high / low) - 1) * 100
        adr_series = daily_range_pct.rolling(20).mean()
        if not adr_series.empty:
            adr = round(adr_series.iloc[-1], 2)
    
    return sma50, sma200, sma10w, sma30w, adr


# ====================== ORIGINAL FUNCTIONS (UNCHANGED) ======================
def log_missing_rs(ticker: str, message: str, log_path: str):
    """Append a debug line to the single Missing_RS.log file"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ticker}] {message}\n")


def align_series(closes, closes_ref):
    return pd.DataFrame({
        "stock": closes,
        "ref": closes_ref
    }).dropna().sort_index()


def debug_alignment(ticker, closes, closes_ref, df, log_path):
    log_missing_rs(ticker,
        f"ALIGNMENT → stock={len(closes)}, ref={len(closes_ref)}, aligned={len(df)}",
        log_path
    )
    if len(df) < min(len(closes), len(closes_ref)) * 0.9:
        log_missing_rs(ticker, "⚠️ ALIGNMENT WARNING: Significant date mismatch!", log_path)
    if len(df) > 5:
        tail_dates = [d.strftime("%Y-%m-%d") for d in df.index[-5:]]
        log_missing_rs(ticker, f"Last aligned dates: {tail_dates}", log_path)


def debug_returns(ticker, df, days, label, log_path, ref_ticker="^CRSLDX"):
    if len(df) < days + 1:
        log_missing_rs(ticker, f"{label} → INSUFFICIENT DATA", log_path)
        return None, None
    
    old_date = df.index[-days-1]
    new_date = df.index[-1]
    s_old = df["stock"].iloc[-days-1]
    s_new = df["stock"].iloc[-1]
    r_old = df["ref"].iloc[-days-1]
    r_new = df["ref"].iloc[-1]

    s_ret = s_new / s_old - 1
    r_ret = r_new / r_old - 1

    log_missing_rs(
        ticker,
        f"{label:>2} → {old_date.date()} → {new_date.date()} | "
        f"Stock: {s_old:.2f} → {s_new:.2f} ({s_ret:+6.2%}) | "
        f"{ref_ticker}: {r_old:.2f} → {r_new:.2f} ({r_ret:+6.2%})",
        log_path
    )
    return s_ret, r_ret


def validate_rs(ticker, rs, s_ret, r_ret, label, log_path):
    if s_ret is None or r_ret is None or pd.isna(rs):
        return
    if s_ret > r_ret and rs < 100:
        log_missing_rs(ticker, f"⚠️ {label} INCONSISTENT: Stock > Ref but RS < 100", log_path)
    if s_ret < r_ret and rs > 100:
        log_missing_rs(ticker, f"⚠️ {label} INCONSISTENT: Stock < Ref but RS > 100", log_path)
    if abs(s_ret) > 2 or abs(r_ret) > 2:
        log_missing_rs(ticker, f"⚠️ {label} EXTREME MOVE (>200%) — possible bad data", log_path)


def debug_trend(ticker, rs_1m, rs_3m, rs_6m, log_path):
    if pd.notna(rs_1m) and pd.notna(rs_3m) and pd.notna(rs_6m):
        if rs_1m > rs_3m > rs_6m:
            log_missing_rs(ticker, "Trend: Accelerating 🚀", log_path)
        elif rs_1m < rs_3m < rs_6m:
            log_missing_rs(ticker, "Trend: Decelerating 📉", log_path)
        else:
            log_missing_rs(ticker, "Trend: Mixed", log_path)


def quarters_perf(closes: pd.Series, n: int) -> float:
    days = n * 63
    slice_len = min(len(closes), days + 1)
    available_data = closes[-slice_len:]
    if len(available_data) < 2:
        return 0.0 if len(available_data) == 1 else np.nan
    pct_change = available_data.pct_change(fill_method=None).dropna()
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
    return round(rs, 2) if rs <= 700 else 700.0


def short_relative_strength(closes: pd.Series, closes_ref: pd.Series, days: int) -> float:
    if len(closes) < days + 1 or len(closes_ref) < days + 1:
        return np.nan

    price_old = closes.iloc[-days - 1]
    price_new = closes.iloc[-1]
    ref_old = closes_ref.iloc[-days - 1]
    ref_new = closes_ref.iloc[-1]

    if price_new <= 0 or ref_new <= 0 or price_old <= 0 or ref_old <= 0:
        return np.nan
    if pd.isna(price_old) or pd.isna(price_new) or pd.isna(ref_old) or pd.isna(ref_new):
        return np.nan

    stock_ret = price_new / price_old - 1
    ref_ret = ref_new / ref_old - 1

    if ref_ret == 0:
        return np.nan if stock_ret <= 0 else 999.0

    rs = (1 + stock_ret) / (1 + ref_ret) * 100
    return round(rs, 2) if rs <= 700 else 700.0


def load_arctic_db(data_dir):
    try:
        if not os.path.exists(data_dir):
            raise Exception(f"ArcticDB directory {data_dir} does not exist")
        arctic = adb.Arctic(f"lmdb://{data_dir}")
        if not arctic.has_library("prices"):
            raise Exception(f"No 'prices' library found in {data_dir}")
        lib = arctic.get_library("prices")
        symbols = lib.list_symbols()
        logging.info(f"Found {len(symbols)} symbols in {data_dir}")
        return lib, symbols
    except Exception as e:
        logging.error(f"Database error in {data_dir}: {str(e)}")
        print(f"ArcticDB error in {data_dir}: {str(e)}")
        return None, None


def generate_tradingview_csv(df_stocks, output_dir, ref_data, percentile_values=None, use_trading_days=True):
    if percentile_values is None:
        percentile_values = [98, 89, 69, 49, 29, 9, 1]

    latest_ts = ref_data["datetime"].max()
    latest_date = datetime.fromtimestamp(latest_ts).date()
    logging.info(f"Latest market date (NSE): {latest_date}")

    dates = []
    if use_trading_days and get_calendar:
        for cal_name in ['NSE', 'XBOM']:
            try:
                cal = get_calendar(cal_name)
                sched = cal.schedule(start_date=latest_date - timedelta(days=20),
                                   end_date=latest_date + timedelta(days=2))
                valid_dates = [d.date() for d in sched.index if d.date() <= latest_date]
                dates = [d.strftime('%Y%m%dT') for d in valid_dates[-5:]]
                if len(dates) >= 5:
                    logging.info(f"NSE trading days used: {', '.join(dates)}")
                    break
            except Exception as e:
                logging.warning(f"{cal_name} calendar failed: {e}")

    if len(dates) < 5:
        dates = [(latest_date - timedelta(days=i)).strftime('%Y%m%dT') for i in range(4, -1, -1)]
        logging.info(f"Fallback consecutive dates: {', '.join(dates)}")

    valid_rs = df_stocks["RS"].dropna().sort_values(ascending=False).reset_index(drop=True)
    total = len(valid_rs)
    rs_map = {}
    for p in percentile_values:
        if total == 0:
            rs_map[p] = 100.0
            continue
        top_n = max(1, round(total * (100 - p) / 100.0))
        threshold_rs = valid_rs.iloc[min(top_n - 1, total - 1)]
        rs_map[p] = round(float(threshold_rs), 2)

    lines = []
    for p in sorted(percentile_values, reverse=True):
        rs_val = rs_map[p]
        for d in dates:
            lines.append(f"{d},0,1000,0,{rs_val},0\n")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "RSRATING.csv")
    with open(path, "w") as f:
        f.write(''.join(lines))

    logging.info(f"RSRATING.csv (NSE) generated → {path}")
    print("=== NSE RSRATING.csv thresholds ===")
    for p in sorted(percentile_values, reverse=True):
        print(f"  {p:2}th percentile → Raw RS ≥ {rs_map[p]:6.2f}")

    return ''.join(lines)


# ====================== MAIN FUNCTION ======================
def main(arctic_db_path, reference_ticker, output_dir, log_file, metadata_file=None, percentiles=None, debug=False):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting NSE India RS calculation with Technical Indicators")

    debug_rs_dir = os.path.join(os.path.dirname(log_file), "debug_rs")
    os.makedirs(debug_rs_dir, exist_ok=True)
    missing_rs_log = os.path.join(debug_rs_dir, "Missing_RS.log")
    open(missing_rs_log, "w").close()
    logging.info(f"Missing RS debug log: {missing_rs_log}")

    result = load_arctic_db(arctic_db_path)
    if not result:
        logging.error("Failed to load ArcticDB. Exiting.")
        print("Failed to load ArcticDB. See logs.")
        sys.exit(1)

    lib, tickers = result

    if reference_ticker not in tickers:
        logging.error(f"Reference ticker {reference_ticker} not found")
        print(f"Reference ticker {reference_ticker} not found in ArcticDB.")
        sys.exit(1)

    ref_data = lib.read(reference_ticker).data
    ref_closes = pd.Series(ref_data["close"].values, index=pd.to_datetime(ref_data["datetime"], unit='s')).sort_index()
    if len(ref_closes) < 20:
        logging.error(f"Reference ticker {reference_ticker} has insufficient data")
        print("Not enough reference ticker data.")
        sys.exit(1)

    # Metadata loading (unchanged)
    metadata_df = pd.DataFrame()
    if metadata_file and os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
            records = []
            if isinstance(data, list):
                for item in data:
                    info = item.get("info", {})
                    records.append({
                        "Ticker": item.get("ticker"),
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
            metadata_df = pd.DataFrame([r for r in records if r["Ticker"]])
        except Exception as e:
            logging.error(f"Invalid metadata file: {e}")

    print(f"Processing {len(tickers)-1:,} Indian stocks...")

    rs_results = []
    valid_rs_count = 0

    for ticker in tqdm(tickers, desc="Calculating RS + Indicators"):
        if ticker == reference_ticker:
            continue

        try:
            data_obj = lib.read(ticker)
            df_data = data_obj.data

            # Original closes for RS calculation
            closes = pd.Series(df_data["close"].values, 
                             index=pd.to_datetime(df_data["datetime"], unit='s')).sort_index()

            log_missing_rs(ticker, f"=== Debug for {ticker} ===", missing_rs_log)
            log_missing_rs(ticker, f"Rows: {len(closes)} | Start={closes.index[0].date() if len(closes)>0 else 'N/A'} | End={closes.index[-1].date() if len(closes)>0 else 'N/A'}", missing_rs_log)

            if len(closes) < 2:
                log_missing_rs(ticker, "NOT ENOUGH DATA (<2 rows)", missing_rs_log)
                rs_results.append((ticker, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue

            # === PREPARE DATAFRAME FOR YOUR INDICATOR FUNCTION ===
            price_df = pd.DataFrame({
                'close': df_data['close']
            }, index=pd.to_datetime(df_data["datetime"], unit='s'))

            for col in ['high', 'low', 'open']:
                if col in df_data.columns:
                    price_df[col] = df_data[col]

            price_df = price_df.sort_index()

            # Call your function
            sma50, sma200, sma10w, sma30w, adr = calculate_smas_and_adr(price_df)

            # === ORIGINAL RS CALCULATION (UNCHANGED) ===
            rs = relative_strength(closes, ref_closes)

            df_aligned = align_series(closes, ref_closes)
            debug_alignment(ticker, closes, ref_closes, df_aligned, missing_rs_log)

            s1, r1 = debug_returns(ticker, df_aligned, 21, "1M", missing_rs_log, reference_ticker)
            s3, r3 = debug_returns(ticker, df_aligned, 63, "3M", missing_rs_log, reference_ticker)
            s6, r6 = debug_returns(ticker, df_aligned, 126, "6M", missing_rs_log, reference_ticker)

            rs_1m = short_relative_strength(closes, ref_closes, 21)
            rs_3m = short_relative_strength(closes, ref_closes, 63)
            rs_6m = short_relative_strength(closes, ref_closes, 126)

            validate_rs(ticker, rs_1m, s1, r1, "1M", missing_rs_log)
            validate_rs(ticker, rs_3m, s3, r3, "3M", missing_rs_log)
            validate_rs(ticker, rs_6m, s6, r6, "6M", missing_rs_log)
            debug_trend(ticker, rs_1m, rs_3m, rs_6m, missing_rs_log)

            log_missing_rs(ticker, f"FINAL → RS={rs}, 1M={rs_1m}, 3M={rs_3m}, 6M={rs_6m} | SMA50={sma50}, SMA200={sma200}, ADR={adr}", missing_rs_log)
            log_missing_rs(ticker, "-" * 60, missing_rs_log)

            rs_results.append((ticker, rs, rs_1m, rs_3m, rs_6m, sma50, sma200, sma10w, sma30w, adr))
            if not np.isnan(rs):
                valid_rs_count += 1

        except Exception as e:
            log_missing_rs(ticker, f"EXCEPTION: {e}", missing_rs_log)
            log_missing_rs(ticker, "-" * 60, missing_rs_log)
            rs_results.append((ticker, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))

    # ====================== OUTPUT ======================
    # ====================== OUTPUT (UNCHANGED) ======================
    df_stocks = pd.DataFrame(rs_results, columns=[
        "Ticker", "RS", "1M_RS", "3M_RS", "6M_RS",
        "SMA50", "SMA200", "SMA10_Weekly", "SMA30_Weekly", "ADR"
    ])

    if not metadata_df.empty:
        df_stocks = df_stocks.merge(metadata_df, on="Ticker", how="left")

    for col in ["RS", "1M_RS", "3M_RS", "6M_RS"]:
        valid = df_stocks[col].dropna()
        if not valid.empty:
            df_stocks.loc[valid.index, f"{col} Percentile"] = (valid.rank(pct=True, method='min') * 99).astype(int)

    df_stocks = df_stocks.sort_values("RS", ascending=False, na_position="last").reset_index(drop=True)
    df_stocks["Rank"] = df_stocks.index + 1
    df_stocks["IPO"] = "No"
    if "Type" in df_stocks.columns:
        df_stocks.loc[df_stocks["Type"] == "ETF", ["Sector", "Industry"]] = "ETF"

    os.makedirs(output_dir, exist_ok=True)

    output_columns = [
        "Rank", "Ticker", "Price", "DVol", "Sector", "Industry",
        "RS Percentile", "1M_RS Percentile", "3M_RS Percentile", "6M_RS Percentile",
        "SMA50", "SMA200", "SMA10_Weekly", "SMA30_Weekly", "ADR",
        "AvgVol", "AvgVol10", "52WKH", "52WKL", "MCAP", "IPO"
    ]

    available_cols = [col for col in output_columns if col in df_stocks.columns]
    df_stocks[available_cols].to_csv(os.path.join(output_dir, "rs_stocks.csv"), index=False, na_rep="")

    # Industry table (unchanged)
    df_industries = df_stocks.groupby("Industry").agg({
        "RS Percentile": "mean", "1M_RS Percentile": "mean",
        "3M_RS Percentile": "mean", "6M_RS Percentile": "mean",
        "Sector": "first",
        "Ticker": lambda x: ",".join(df_stocks[df_stocks["Ticker"].isin(x)].sort_values("RS", ascending=False)["Ticker"])
    }).reset_index()

    for col in ["RS Percentile", "1M_RS Percentile", "3M_RS Percentile", "6M_RS Percentile"]:
        df_industries[col] = df_industries[col].fillna(0).round().astype(int)
    df_industries = df_industries.sort_values("RS Percentile", ascending=False).reset_index(drop=True)
    df_industries["Rank"] = df_industries.index + 1
    df_industries.rename(columns={
        "RS Percentile": "RS", "1M_RS Percentile": "1 M_RS",
        "3M_RS Percentile": "3M_RS", "6M_RS Percentile": "6M_RS"
    }, inplace=True)
    df_industries[["Rank", "Industry", "Sector", "RS", "1 M_RS", "3M_RS", "6M_RS", "Ticker"]].to_csv(
        os.path.join(output_dir, "rs_industries.csv"), index=False)

    generate_tradingview_csv(df_stocks, output_dir, ref_data, percentiles)

    logging.info(f"NSE RS calculation completed. {len(df_stocks)} tickers, {valid_rs_count} with valid RS.")
    print(f"\n✅ ENHANCED NSE RS CALCULATION COMPLETE!")
    print(f"Valid RS: {valid_rs_count:,} / {len(df_stocks):,}")
    print(f"Output → {output_dir}/")
    print(f"   • rs_stocks.csv (with SMA50, SMA200, SMA10_Weekly, SMA30_Weekly, ADR)")
    print(f"   • rs_industries.csv")
    print(f"   • RSRATING.csv")
    print(f"   • logs/debug_rs/Missing_RS.log")

    if debug:
        print("\nStarting FULL DEBUG export...")
        debug_records = []
        strength_ref_cached = strength(ref_closes)
        for row in tqdm(df_stocks.itertuples(), total=len(df_stocks), desc="Debug Export"):
            ticker = row.Ticker
            try:
                data = lib.read(ticker).data
                closes = pd.Series(data["close"].values, index=pd.to_datetime(data["datetime"], unit='s'))
                num_days = len(closes)
                perf_3m = quarters_perf(closes, 1)
                perf_6m = quarters_perf(closes, 2)
                perf_9m = quarters_perf(closes, 3)
                perf_12m = quarters_perf(closes, 4)
                strength_stock = strength(closes)
                rs = getattr(row, 'RS', np.nan)
                recomputed = relative_strength(closes, ref_closes)

                debug_records.append({
                    'Ticker': ticker, 'Days': num_days,
                    '3M_Return': round(perf_3m, 4) if not np.isnan(perf_3m) else None,
                    '6M_Return': round(perf_6m, 4) if not np.isnan(perf_6m) else None,
                    '9M_Return': round(perf_9m, 4) if not np.isnan(perf_9m) else None,
                    '12M_Return': round(perf_12m, 4) if not np.isnan(perf_12m) else None,
                    'Strength_Stock': round(strength_stock, 4),
                    'Strength_Ref': round(strength_ref_cached, 4),
                    'RS': rs, 'Recomputed_RS': recomputed,
                    'Match': "OK" if abs((rs or 0) - (recomputed or 0)) < 0.1 else "MISMATCH"
                })
            except:
                debug_records.append({'Ticker': ticker, 'Days': 0, 'Match': 'Error'})

        for i, chunk in enumerate([debug_records[x:x+3990] for x in range(0, len(debug_records), 3990)]):
            pd.DataFrame(chunk).to_csv(os.path.join(debug_rs_dir, f"debug-rs-part{i+1}.csv"), index=False)
        print(f"Full debug export done → {debug_rs_dir}/")
        print("\nStarting FULL DEBUG export... (original logic)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate RS Rating for NSE India vs Benchmark + Technical Indicators")
    parser.add_argument("--arctic-db-path", default="tmp/arctic_db", help="Path to ArcticDB")
    parser.add_argument("--reference-ticker", default="^CRSLDX", help="Benchmark index")
    parser.add_argument("--output-dir", default="RS_Data", help="Output directory")
    parser.add_argument("--log-file", default="logs/calc.log", help="Log file")
    parser.add_argument("--metadata-file", default="data/ticker_price.json", help="Metadata JSON")
    parser.add_argument("--percentiles", default="98,89,69,49,29,9,1", help="Percentiles for RSRATING.csv")
    parser.add_argument("--debug", action="store_true", help="Enable full debug export")
    args = parser.parse_args()

    percentiles = [int(p) for p in args.percentiles.split(",")]
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    main(args.arctic_db_path, args.reference_ticker, args.output_dir, args.log_file,
         args.metadata_file, percentiles, args.debug)
