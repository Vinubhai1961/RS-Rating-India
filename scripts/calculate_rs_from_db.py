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
    logging.warning(
        "pandas_market_calendars not installed. "
        "Falling back to consecutive days for RSRATING.csv."
    )


# ====================== Unified Missing RS Logger ======================
def log_missing_rs(ticker: str, message: str, log_path: str):
    """Append a debug line to the single Missing_RS.log file."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ticker}] {message}\n")


# ====================== ENHANCED RS DEBUG ======================
def align_series(closes, closes_ref):
    return pd.DataFrame({
        "stock": closes,
        "ref": closes_ref
    }).dropna().sort_index()


def debug_alignment(ticker, closes, closes_ref, df, log_path):
    log_missing_rs(
        ticker,
        f"ALIGNMENT → stock={len(closes)}, ref={len(closes_ref)}, aligned={len(df)}",
        log_path
    )

    if len(df) < min(len(closes), len(closes_ref)) * 0.9:
        log_missing_rs(
            ticker,
            "⚠️ ALIGNMENT WARNING: Significant date mismatch!",
            log_path
        )

    if len(df) > 5:
        tail_dates = [d.strftime("%Y-%m-%d") for d in df.index[-5:]]
        log_missing_rs(
            ticker,
            f"Last aligned dates: {tail_dates}",
            log_path
        )


def debug_returns(ticker, df, days, label, log_path, ref_ticker="^CRSLDX"):
    if len(df) < days + 1:
        log_missing_rs(
            ticker,
            f"{label} → INSUFFICIENT DATA",
            log_path
        )
        return None, None

    old_date = df.index[-days - 1]
    new_date = df.index[-1]
    s_old = df["stock"].iloc[-days - 1]
    s_new = df["stock"].iloc[-1]
    r_old = df["ref"].iloc[-days - 1]
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
        log_missing_rs(
            ticker,
            f"⚠️ {label} INCONSISTENT: Stock > Ref but RS < 100",
            log_path
        )

    if s_ret < r_ret and rs > 100:
        log_missing_rs(
            ticker,
            f"⚠️ {label} INCONSISTENT: Stock < Ref but RS > 100",
            log_path
        )

    if abs(s_ret) > 2 or abs(r_ret) > 2:
        log_missing_rs(
            ticker,
            f"⚠️ {label} EXTREME MOVE (>200%) — possible bad data",
            log_path
        )


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

    return (
        (pct_change + 1).cumprod().iloc[-1] - 1
        if not pct_change.empty
        else np.nan
    )


def strength(closes: pd.Series) -> float:
    perfs = [quarters_perf(closes, i) for i in range(1, 5)]
    valid_perfs = [p for p in perfs if not np.isnan(p)]

    if not valid_perfs:
        return np.nan

    weights = [0.4, 0.2, 0.2, 0.2][:len(valid_perfs)]
    total_weight = sum(weights)

    weights = [
        w / total_weight
        for w in weights
    ] if total_weight > 0 else weights

    return sum(w * p for w, p in zip(weights, valid_perfs))


def relative_strength(closes: pd.Series, closes_ref: pd.Series) -> float:
    rs_stock = strength(closes)
    rs_ref = strength(closes_ref)

    if np.isnan(rs_stock) or np.isnan(rs_ref):
        logging.info(
            f"NaN RS for ticker with {len(closes)} days, "
            f"ref with {len(closes_ref)} days"
        )
        return np.nan

    rs = (1 + rs_stock) / (1 + rs_ref) * 100

    return round(rs, 2) if rs <= 700 else 700.0


def short_relative_strength(
    closes: pd.Series,
    closes_ref: pd.Series,
    days: int
) -> float:
    """Date-aligned short RS calculation."""
    if len(closes) < days + 5 or len(closes_ref) < days + 5:
        return np.nan

    df = pd.DataFrame({
        "stock": closes,
        "ref": closes_ref
    }).dropna().sort_index()

    if len(df) < days + 1:
        return np.nan

    price_old = df["stock"].iloc[-days - 1]
    price_new = df["stock"].iloc[-1]
    ref_old = df["ref"].iloc[-days - 1]
    ref_new = df["ref"].iloc[-1]

    if price_new <= 0 or ref_new <= 0 or price_old <= 0 or ref_old <= 0:
        return np.nan

    if (
        pd.isna(price_old)
        or pd.isna(price_new)
        or pd.isna(ref_old)
        or pd.isna(ref_new)
    ):
        return np.nan

    stock_ret = price_new / price_old - 1
    ref_ret = ref_new / ref_old - 1

    if abs(ref_ret) < 0.0001:
        return np.nan if stock_ret <= 0 else 999.0

    rs = (1 + stock_ret) / (1 + ref_ret) * 100

    return round(rs, 2) if rs <= 700 else 700.0


# ====================== NEW: SMA CALCULATION HELPER ======================
def calculate_smas(closes: pd.Series):
    """
    Calculate daily and weekly moving averages:
      - SMA50
      - SMA200
      - SMA10W
      - SMA30W
    """
    sma50 = np.nan
    sma200 = np.nan
    sma10w = np.nan
    sma30w = np.nan

    closes = closes.dropna().sort_index()

    if len(closes) >= 50:
        sma50 = round(float(closes.rolling(window=50).mean().iloc[-1]), 2)

    if len(closes) >= 200:
        sma200 = round(float(closes.rolling(window=200).mean().iloc[-1]), 2)

    weekly_closes = closes.resample("W").last().dropna()

    if len(weekly_closes) >= 10:
        sma10w = round(float(weekly_closes.rolling(window=10).mean().iloc[-1]), 2)

    if len(weekly_closes) >= 30:
        sma30w = round(float(weekly_closes.rolling(window=30).mean().iloc[-1]), 2)

    return sma50, sma200, sma10w, sma30w


# ====================== NEW: ATR & ADR HELPER ======================
def get_atr_adr(ticker: str, lib, period: int = 14):
    """
    Calculate latest ATR(period) and ADR(period).

    ATR uses True Range:
      max(high-low, abs(high-prev_close), abs(low-prev_close))

    ADR uses average high-low range.
    """
    try:
        data = lib.read(ticker).data

        required_cols = {"high", "low", "close", "datetime"}
        if not required_cols.issubset(set(data.columns)):
            missing = required_cols - set(data.columns)
            logging.warning(f"ATR/ADR skipped for {ticker}; missing columns: {missing}")
            return np.nan, np.nan

        df = pd.DataFrame({
            "high": data["high"].values,
            "low": data["low"].values,
            "close": data["close"].values,
        }, index=pd.to_datetime(data["datetime"], unit="s")).sort_index()

        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["high", "low", "close"]
        )

        df = df[
            (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
        ]

        if len(df) < period + 1:
            return np.nan, np.nan

        tr0 = df["high"] - df["low"]
        tr1 = (df["high"] - df["close"].shift(1)).abs()
        tr2 = (df["low"] - df["close"].shift(1)).abs()

        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean().iloc[-1]
        adr = (df["high"] - df["low"]).rolling(window=period).mean().iloc[-1]

        return round(float(atr), 4), round(float(adr), 4)

    except Exception as e:
        logging.warning(f"ATR/ADR calculation failed for {ticker}: {e}")
        return np.nan, np.nan



def calculate_atr_adr_from_dataframe(data: pd.DataFrame, ticker: str, period: int = 14):
    """
    Calculate latest ATR(period) and ADR(period) from an already-loaded
    ArcticDB dataframe.

    This preserves the original ATR/ADR formula but avoids a second
    lib.read(ticker) call for every symbol.
    """
    try:
        required_cols = {"high", "low", "close", "datetime"}

        if not required_cols.issubset(set(data.columns)):
            missing = required_cols - set(data.columns)
            logging.warning(f"ATR/ADR skipped for {ticker}; missing columns: {missing}")
            return np.nan, np.nan

        df = pd.DataFrame({
            "high": data["high"].values,
            "low": data["low"].values,
            "close": data["close"].values,
        }, index=pd.to_datetime(data["datetime"], unit="s")).sort_index()

        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["high", "low", "close"]
        )

        df = df[
            (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
        ]

        if len(df) < period + 1:
            return np.nan, np.nan

        tr0 = df["high"] - df["low"]
        tr1 = (df["high"] - df["close"].shift(1)).abs()
        tr2 = (df["low"] - df["close"].shift(1)).abs()

        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean().iloc[-1]
        adr = (df["high"] - df["low"]).rolling(window=period).mean().iloc[-1]

        return round(float(atr), 4), round(float(adr), 4)

    except Exception as e:
        logging.warning(f"ATR/ADR calculation failed for {ticker}: {e}")
        return np.nan, np.nan


def safe_float(value):
    if value is None or value == "":
        return np.nan

    try:
        if isinstance(value, str):
            value = value.strip()

            if value.endswith("K"):
                return float(value[:-1]) * 1_000
            if value.endswith("M"):
                return float(value[:-1]) * 1_000_000
            if value.endswith("B"):
                return float(value[:-1]) * 1_000_000_000

        return float(value)

    except Exception:
        return np.nan


def load_metadata(metadata_file):
    """
    Robust metadata loader for NSE.

    Supports:
      1. list format:
         [{"ticker": "...", "info": {...}}, ...]

      2. dict format:
         {"TCS.NS": {"info": {...}}, ...}

    Preserves NSE field casing:
      Sector, Industry, Type/type, Price, DVol, AvgVol, AvgVol10, 52WKH, 52WKL, MCAP
    """
    metadata_df = pd.DataFrame()

    if not metadata_file or not os.path.exists(metadata_file):
        logging.warning(f"Metadata file not found or not provided: {metadata_file}")
        return metadata_df

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []

        if isinstance(data, list):
            for item in data:
                info = item.get("info", {}) or {}
                ticker = item.get("ticker")

                if not ticker:
                    continue

                records.append({
                    "Ticker": ticker,
                    "Price": info.get("Price"),
                    "DVol": info.get("DVol", ""),
                    "Sector": info.get("Sector", info.get("sector", "")),
                    "Industry": info.get("Industry", info.get("industry", "")),
                    "AvgVol": info.get("AvgVol", ""),
                    "AvgVol10": info.get("AvgVol10", ""),
                    "52WKH": info.get("52WKH"),
                    "52WKL": info.get("52WKL"),
                    "MCAP": info.get("MCAP"),
                    "Type": info.get("type", info.get("Type", "Stock"))
                })

        elif isinstance(data, dict):
            for ticker, payload in data.items():
                info = (payload or {}).get("info", {}) or {}

                records.append({
                    "Ticker": ticker,
                    "Price": info.get("Price"),
                    "DVol": info.get("DVol", ""),
                    "Sector": info.get("Sector", info.get("sector", "")),
                    "Industry": info.get("Industry", info.get("industry", "")),
                    "AvgVol": info.get("AvgVol", ""),
                    "AvgVol10": info.get("AvgVol10", ""),
                    "52WKH": info.get("52WKH"),
                    "52WKL": info.get("52WKL"),
                    "MCAP": info.get("MCAP"),
                    "Type": info.get("type", info.get("Type", "Stock"))
                })

        else:
            raise ValueError(f"Unsupported metadata format: {type(data).__name__}")

        metadata_df = pd.DataFrame(records)

        if metadata_df.empty or "Ticker" not in metadata_df.columns:
            logging.warning(f"Metadata file {metadata_file} produced no valid records.")
            return pd.DataFrame()

        metadata_df = metadata_df.drop_duplicates(subset=["Ticker"], keep="first")

        logging.info(f"Metadata loaded: {len(metadata_df):,} tickers")

        return metadata_df

    except Exception as e:
        logging.error(f"Invalid metadata file {metadata_file}: {str(e)}")
        return pd.DataFrame()


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


def generate_tradingview_csv(
    df_stocks,
    output_dir,
    ref_data,
    percentile_values=None,
    use_trading_days=True
):
    if percentile_values is None:
        percentile_values = [98, 89, 69, 49, 29, 9, 1]

    latest_ts = ref_data["datetime"].max()
    latest_date = datetime.fromtimestamp(latest_ts).date()

    logging.info(f"Latest market date (NSE): {latest_date}")

    dates = []

    if use_trading_days and get_calendar:
        for cal_name in ["NSE", "XBOM"]:
            try:
                cal = get_calendar(cal_name)

                sched = cal.schedule(
                    start_date=latest_date - timedelta(days=20),
                    end_date=latest_date + timedelta(days=2)
                )

                valid_dates = [
                    d.date()
                    for d in sched.index
                    if d.date() <= latest_date
                ]

                dates = [
                    d.strftime("%Y%m%dT")
                    for d in valid_dates[-5:]
                ]

                if len(dates) >= 5:
                    logging.info(
                        f"{cal_name} trading days used: "
                        f"{', '.join(dates)}"
                    )
                    break

            except Exception as e:
                logging.warning(f"{cal_name} calendar failed: {e}")

    if len(dates) < 5:
        dates = [
            (latest_date - timedelta(days=i)).strftime("%Y%m%dT")
            for i in range(4, -1, -1)
        ]

        logging.info(
            f"Fallback consecutive dates: {', '.join(dates)}"
        )

    valid_rs = (
        df_stocks["RS"]
        .dropna()
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )

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

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    logging.info(f"RSRATING.csv (NSE) generated → {path}")

    print("=== NSE RSRATING.csv thresholds ===")

    for p in sorted(percentile_values, reverse=True):
        print(f"  {p:2}th percentile → Raw RS ≥ {rs_map[p]:6.2f}")

    return "".join(lines)

def build_rs_threshold_map(df_stocks, column, percentile_values):
    valid_rs = (
        df_stocks[column]
        .dropna()
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )

    total = len(valid_rs)
    rs_map = {}

    for p in percentile_values:
        if total == 0:
            rs_map[p] = 100.0
            continue

        top_n = max(1, round(total * (100 - p) / 100.0))
        threshold_rs = valid_rs.iloc[min(top_n - 1, total - 1)]
        rs_map[p] = round(float(threshold_rs), 2)

    return rs_map


def generate_pine_thresholds(df_stocks, output_dir, percentile_values):
    threshold_sets = {
        "ind": "RS",
        "ind1m": "1M_RS",
        "ind3m": "3M_RS",
        "ind6m": "6M_RS",
    }

    lines = []
    lines.append("// Auto-generated RS Rating thresholds - do not edit manually\n")
    lines.append(f"// Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")

    for prefix, col in threshold_sets.items():
        rs_map = build_rs_threshold_map(df_stocks, col, percentile_values)
        print(f"\n=== {col} Pine thresholds ===")
        
        for p in sorted(percentile_values, reverse=True):
            print(f"  {p:2}th percentile → Raw {col} ≥ {rs_map[p]:6.2f}")
            
        lines.append(f"// {col} thresholds\n")
        
        for p in sorted(percentile_values, reverse=True):
            label = f"{prefix}{p:02d}"
            lines.append(
                f'{label} = input.float({rs_map[p]:.2f}, "{prefix.upper()} {p}th → RS ≥", group="{prefix.upper()} Thresholds")\n'
            )
        lines.append("\n")

    path = os.path.join(output_dir, "RS-Rating-pine.csv")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"RS-Rating-pine.csv Pine thresholds generated → {path}")


def main(
    arctic_db_path,
    reference_ticker,
    output_dir,
    log_file,
    metadata_file=None,
    percentiles=None,
    debug=False
):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )

    logging.info("Starting NSE India RS calculation")

    debug_rs_dir = os.path.join(os.path.dirname(log_file), "debug_rs")
    os.makedirs(debug_rs_dir, exist_ok=True)

    missing_rs_log = os.path.join(debug_rs_dir, "Missing_RS.log")
    open(missing_rs_log, "w", encoding="utf-8").close()

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

    ref_closes = pd.Series(
        ref_data["close"].values,
        index=pd.to_datetime(ref_data["datetime"], unit="s")
    ).sort_index()

    if len(ref_closes) < 20:
        logging.error(
            f"Reference ticker {reference_ticker} has insufficient data "
            f"({len(ref_closes)} days)"
        )
        print("Not enough reference ticker data.")
        sys.exit(1)

    # === ROBUST INDIA METADATA LOADING ===
    metadata_df = load_metadata(metadata_file)

    logging.info(
        f"Starting RS calculation for {len(tickers)} tickers "
        f"vs {reference_ticker}"
    )

    print(f"Processing {len(tickers) - 1:,} Indian stocks...")

    rs_results = []
    valid_rs_count = 0

    # Runtime-only data quality counters.
    # These are printed/logged only and are not committed as a separate file.
    quality_stats = {
        "total_processed": 0,
        "valid_rs": 0,
        "missing_rs": 0,
        "missing_1m_rs": 0,
        "missing_3m_rs": 0,
        "missing_6m_rs": 0,
        "missing_atr": 0,
        "missing_adr": 0,
        "missing_sma50": 0,
        "missing_sma200": 0,
        "short_history_lt_252": 0,
        "exceptions": 0,
    }

    for ticker in tqdm(tickers, desc="Calculating RS"):
        if ticker == reference_ticker:
            continue

        try:
            data = lib.read(ticker).data

            closes = pd.Series(
                data["close"].values,
                index=pd.to_datetime(data["datetime"], unit="s")
            ).sort_index()

            quality_stats["total_processed"] += 1

            if len(closes) < 252:
                quality_stats["short_history_lt_252"] += 1

            log_missing_rs(ticker, f"=== Debug for {ticker} ===", missing_rs_log)

            if len(closes) > 0:
                log_missing_rs(
                    ticker,
                    f"Rows: {len(closes)} | "
                    f"Start={closes.index[0].date()} | "
                    f"End={closes.index[-1].date()}",
                    missing_rs_log
                )
            else:
                log_missing_rs(ticker, "Rows: 0", missing_rs_log)

            log_missing_rs(
                ticker,
                f"Has_1M={len(closes) >= 22}, "
                f"Has_3M={len(closes) >= 64}, "
                f"Has_6M={len(closes) >= 127}, "
                f"Has_12M={len(closes) >= 253}",
                missing_rs_log
            )

            if len(closes) < 2:
                log_missing_rs(
                    ticker,
                    "NOT ENOUGH DATA (<2 rows)",
                    missing_rs_log
                )
                rs_results.append((
                    ticker,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan
                ))
                continue

            # === NEW: SMA ===
            sma50, sma200, sma10w, sma30w = calculate_smas(closes)

            # Main RS
            rs = relative_strength(closes, ref_closes)

            if pd.isna(rs):
                log_missing_rs(
                    ticker,
                    f"RS=NaN → stock_strength={strength(closes):.4f}, "
                    f"ref_strength={strength(ref_closes):.4f}",
                    missing_rs_log
                )
            else:
                log_missing_rs(ticker, f"RS={rs}", missing_rs_log)

            # === NEW: ATR / ADR ===
            # Use already-loaded dataframe to avoid a second ArcticDB read per ticker.
            atr, adr = calculate_atr_adr_from_dataframe(data, ticker, period=14)

            # ====================== ENHANCED DEBUG BLOCK ======================
            df_aligned = align_series(closes, ref_closes)

            debug_alignment(
                ticker,
                closes,
                ref_closes,
                df_aligned,
                missing_rs_log
            )

            s1, r1 = debug_returns(
                ticker,
                df_aligned,
                21,
                "1M",
                missing_rs_log,
                reference_ticker
            )

            s3, r3 = debug_returns(
                ticker,
                df_aligned,
                63,
                "3M",
                missing_rs_log,
                reference_ticker
            )

            s6, r6 = debug_returns(
                ticker,
                df_aligned,
                126,
                "6M",
                missing_rs_log,
                reference_ticker
            )

            # Original short RS calculation preserved.
            rs_1m = short_relative_strength(closes, ref_closes, 21)
            rs_3m = short_relative_strength(closes, ref_closes, 63)
            rs_6m = short_relative_strength(closes, ref_closes, 126)

            validate_rs(ticker, rs_1m, s1, r1, "1M", missing_rs_log)
            validate_rs(ticker, rs_3m, s3, r3, "3M", missing_rs_log)
            validate_rs(ticker, rs_6m, s6, r6, "6M", missing_rs_log)

            debug_trend(ticker, rs_1m, rs_3m, rs_6m, missing_rs_log)
            # ====================== END ENHANCED DEBUG ======================

            log_missing_rs(
                ticker,
                f"FINAL → RS={rs}, 1M={rs_1m}, 3M={rs_3m}, 6M={rs_6m} | "
                f"SMA50={sma50}, SMA200={sma200}, "
                f"SMA10W={sma10w}, SMA30W={sma30w}, "
                f"ATR={atr}, ADR={adr}",
                missing_rs_log
            )

            log_missing_rs(ticker, "-" * 60, missing_rs_log)

            rs_results.append((
                ticker,
                rs,
                rs_1m,
                rs_3m,
                rs_6m,
                sma50,
                sma200,
                sma10w,
                sma30w,
                atr,
                adr
            ))

            if not np.isnan(rs):
                valid_rs_count += 1
                quality_stats["valid_rs"] += 1
            else:
                quality_stats["missing_rs"] += 1

            if pd.isna(rs_1m):
                quality_stats["missing_1m_rs"] += 1
            if pd.isna(rs_3m):
                quality_stats["missing_3m_rs"] += 1
            if pd.isna(rs_6m):
                quality_stats["missing_6m_rs"] += 1
            if pd.isna(atr):
                quality_stats["missing_atr"] += 1
            if pd.isna(adr):
                quality_stats["missing_adr"] += 1
            if pd.isna(sma50):
                quality_stats["missing_sma50"] += 1
            if pd.isna(sma200):
                quality_stats["missing_sma200"] += 1

        except Exception as e:
            quality_stats["exceptions"] += 1

            log_missing_rs(
                ticker,
                f"EXCEPTION: {e}",
                missing_rs_log
            )

            log_missing_rs(
                ticker,
                "-" * 60,
                missing_rs_log
            )

            rs_results.append((
                ticker,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan
            ))

    df_stocks = pd.DataFrame(
        rs_results,
        columns=[
            "Ticker",
            "RS",
            "1M_RS",
            "3M_RS",
            "6M_RS",
            "SMA50",
            "SMA200",
            "SMA10W",
            "SMA30W",
            "ATR",
            "ADR"
        ]
    )

    if not metadata_df.empty and "Ticker" in metadata_df.columns:
        df_stocks = df_stocks.merge(
            metadata_df,
            on="Ticker",
            how="left"
        )

    if df_stocks.empty:
        logging.warning("No tickers processed due to errors or empty data")
        print("No RS results calculated. Check ArcticDB/reference ticker.")
        sys.exit(1)

    # Percentiles.
    for col in ["RS", "1M_RS", "3M_RS", "6M_RS"]:
        valid = df_stocks[col].dropna()

        if not valid.empty:
            df_stocks.loc[
                valid.index,
                f"{col} Percentile"
            ] = (
                valid.rank(pct=True, method="min") * 99
            ).astype(int)
        else:
            df_stocks[f"{col} Percentile"] = np.nan

    df_stocks = (
        df_stocks
        .sort_values("RS", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    df_stocks["Rank"] = df_stocks.index + 1
    df_stocks["IPO"] = "No"

    if "Type" in df_stocks.columns:
        df_stocks.loc[
            df_stocks["Type"] == "ETF",
            ["Sector", "Industry"]
        ] = "ETF"

    os.makedirs(output_dir, exist_ok=True)

    # ====================== ROBUST COLUMN SELECTION ======================
    final_columns = [
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
        "SMA30W"
    ]

    available_cols = [
        col
        for col in final_columns
        if col in df_stocks.columns
    ]

    df_stocks[available_cols].to_csv(
        os.path.join(output_dir, "rs_stocks.csv"),
        index=False,
        na_rep=""
    )

    # ====================== INDUSTRY TABLE ======================
    # Ensure required fields exist before groupby.
    for col in [
        "Industry",
        "Sector",
        "RS Percentile",
        "1M_RS Percentile",
        "3M_RS Percentile",
        "6M_RS Percentile"
    ]:
        if col not in df_stocks.columns:
            df_stocks[col] = np.nan

    df_industries = df_stocks.groupby("Industry", dropna=False).agg({
        "RS Percentile": "mean",
        "1M_RS Percentile": "mean",
        "3M_RS Percentile": "mean",
        "6M_RS Percentile": "mean",
        "Sector": "first",
        "Ticker": lambda x: ",".join(
            df_stocks[
                df_stocks["Ticker"].isin(x)
            ].sort_values(
                "RS",
                ascending=False
            )["Ticker"]
        )
    }).reset_index()

    for col in [
        "RS Percentile",
        "1M_RS Percentile",
        "3M_RS Percentile",
        "6M_RS Percentile"
    ]:
        df_industries[col] = (
            df_industries[col]
            .fillna(0)
            .round()
            .astype(int)
        )

    df_industries = (
        df_industries
        .sort_values("RS Percentile", ascending=False)
        .reset_index(drop=True)
    )

    df_industries["Rank"] = df_industries.index + 1

    df_industries.rename(
        columns={
            "RS Percentile": "RS",
            "1M_RS Percentile": "1 M_RS",
            "3M_RS Percentile": "3M_RS",
            "6M_RS Percentile": "6M_RS"
        },
        inplace=True
    )

    df_industries[
        [
            "Rank",
            "Industry",
            "Sector",
            "RS",
            "1 M_RS",
            "3M_RS",
            "6M_RS",
            "Ticker"
        ]
    ].to_csv(
        os.path.join(output_dir, "rs_industries.csv"),
        index=False
    )

    # ====================== SECTOR TABLE ======================
    # Similar to rs_industries.csv:
    # Rank, Sector, RS, 1 M_RS, 3M_RS, 6M_RS, Ticker
    df_sectors = df_stocks.groupby("Sector", dropna=False).agg({
        "RS Percentile": "mean",
        "1M_RS Percentile": "mean",
        "3M_RS Percentile": "mean",
        "6M_RS Percentile": "mean",
        "Ticker": lambda x: ",".join(
            df_stocks[
                df_stocks["Ticker"].isin(x)
            ].sort_values(
                "RS",
                ascending=False
            )["Ticker"]
        )
    }).reset_index()

    for col in [
        "RS Percentile",
        "1M_RS Percentile",
        "3M_RS Percentile",
        "6M_RS Percentile"
    ]:
        df_sectors[col] = (
            df_sectors[col]
            .fillna(0)
            .round()
            .astype(int)
        )

    df_sectors = (
        df_sectors
        .sort_values("RS Percentile", ascending=False)
        .reset_index(drop=True)
    )

    df_sectors["Rank"] = df_sectors.index + 1

    df_sectors.rename(
        columns={
            "RS Percentile": "RS",
            "1M_RS Percentile": "1 M_RS",
            "3M_RS Percentile": "3M_RS",
            "6M_RS Percentile": "6M_RS"
        },
        inplace=True
    )

    df_sectors[
        [
            "Rank",
            "Sector",
            "RS",
            "1 M_RS",
            "3M_RS",
            "6M_RS",
            "Ticker"
        ]
    ].to_csv(
        os.path.join(output_dir, "rs_sectors.csv"),
        index=False
    )

    # ====================== INDUSTRY LEADERS ======================
    # Top 5 RS stocks within each industry.
    # This is meant for manual review and does not alter rs_stocks.csv.
    leader_source = df_stocks.copy()

    leader_source["Industry"] = leader_source["Industry"].fillna("Unknown")
    leader_source["Sector"] = leader_source["Sector"].fillna("Unknown")

    leader_source = leader_source.dropna(subset=["RS Percentile"])

    industry_leaders = (
        leader_source
        .sort_values(
            [
                "Industry",
                "RS Percentile",
                "3M_RS Percentile",
                "1M_RS Percentile"
            ],
            ascending=[True, False, False, False],
            na_position="last"
        )
        .groupby("Industry", group_keys=False)
        .head(5)
        .copy()
    )

    industry_leaders["IndustryRank"] = (
        industry_leaders
        .groupby("Industry")
        .cumcount() + 1
    )

    leader_columns = [
        "IndustryRank",
        "Ticker",
        "Price",
        "Sector",
        "Industry",
        "RS Percentile",
        "1M_RS Percentile",
        "3M_RS Percentile",
        "6M_RS Percentile",
        "ATR",
        "ADR",
        "AvgVol10",
        "52WKH",
        "52WKL",
        "MCAP",
        "SMA50",
        "SMA200",
        "SMA10W",
        "SMA30W"
    ]

    industry_leader_cols = [
        col for col in leader_columns
        if col in industry_leaders.columns
    ]

    industry_leaders[industry_leader_cols].to_csv(
        os.path.join(output_dir, "industry_leaders.csv"),
        index=False,
        na_rep=""
    )

    generate_tradingview_csv(
        df_stocks,
        output_dir,
        ref_data,
        percentiles
    )
    
    generate_pine_thresholds(
    df_stocks,
    output_dir,
    percentiles
    ) 
    logging.info(
        f"NSE RS calculation completed. "
        f"{len(df_stocks)} tickers, "
        f"{valid_rs_count} with valid RS."
    )

    # ====================== RUNTIME DATA QUALITY SUMMARY ======================
    # Printed/logged only. No summary artifact is written or committed.
    missing_rs_total = len(df_stocks) - valid_rs_count
    missing_atr_total = int(df_stocks["ATR"].isna().sum()) if "ATR" in df_stocks.columns else 0
    missing_adr_total = int(df_stocks["ADR"].isna().sum()) if "ADR" in df_stocks.columns else 0
    missing_sma50_total = int(df_stocks["SMA50"].isna().sum()) if "SMA50" in df_stocks.columns else 0
    missing_sma200_total = int(df_stocks["SMA200"].isna().sum()) if "SMA200" in df_stocks.columns else 0

    data_quality_summary = f"""
================ DATA QUALITY SUMMARY ================
Total ArcticDB symbols processed : {quality_stats["total_processed"]:,}
Rows in rs_stocks.csv            : {len(df_stocks):,}
Valid RS                         : {valid_rs_count:,}
Missing RS                       : {missing_rs_total:,}
Missing 1M_RS                    : {quality_stats["missing_1m_rs"]:,}
Missing 3M_RS                    : {quality_stats["missing_3m_rs"]:,}
Missing 6M_RS                    : {quality_stats["missing_6m_rs"]:,}
Missing ATR                      : {missing_atr_total:,}
Missing ADR                      : {missing_adr_total:,}
Missing SMA50                    : {missing_sma50_total:,}
Missing SMA200                   : {missing_sma200_total:,}
Short history <252 days          : {quality_stats["short_history_lt_252"]:,}
Exceptions                       : {quality_stats["exceptions"]:,}
Industries generated             : {len(df_industries):,}
Sectors generated                : {len(df_sectors):,}
Industry leader rows             : {len(industry_leaders):,}
======================================================
"""

    print(data_quality_summary)
    logging.info(data_quality_summary)

    print("\nNSE RS CALCULATION COMPLETE!")
    print(f"Valid RS: {valid_rs_count:,} / {len(df_stocks):,}")
    print(f"Output → {output_dir}/")
    print("   • rs_stocks.csv")
    print("   • rs_industries.csv")
    print("   • rs_sectors.csv")
    print("   • industry_leaders.csv")
    print("   • RSRATING.csv")
    print("   • logs/debug_rs/Missing_RS.log  (full diagnostics)")

    if debug:
        print("\nStarting FULL DEBUG export...")

        debug_records = []
        strength_ref_cached = strength(ref_closes)

        for row in tqdm(
            df_stocks.itertuples(),
            total=len(df_stocks),
            desc="Debug Export"
        ):
            ticker = row.Ticker

            try:
                data = lib.read(ticker).data

                closes = pd.Series(
                    data["close"].values,
                    index=pd.to_datetime(data["datetime"], unit="s")
                ).sort_index()

                num_days = len(closes)
                start_date = closes.index[0].date() if num_days else None
                end_date = closes.index[-1].date() if num_days else None
                sufficient = (
                    "Sufficient"
                    if num_days >= 252
                    else f"Short: {num_days}d"
                )

                perf_3m = quarters_perf(closes, 1)
                perf_6m = quarters_perf(closes, 2)
                perf_9m = quarters_perf(closes, 3)
                perf_12m = quarters_perf(closes, 4)
                strength_stock = strength(closes)
                rs = getattr(row, "RS", np.nan)
                recomputed = relative_strength(closes, ref_closes)

                debug_records.append({
                    "Ticker": ticker,
                    "Days": num_days,
                    "Start_Date": start_date,
                    "End_Date": end_date,
                    "Sufficient": sufficient,
                    "3M_Return": round(perf_3m, 4) if not np.isnan(perf_3m) else None,
                    "6M_Return": round(perf_6m, 4) if not np.isnan(perf_6m) else None,
                    "9M_Return": round(perf_9m, 4) if not np.isnan(perf_9m) else None,
                    "12M_Return": round(perf_12m, 4) if not np.isnan(perf_12m) else None,
                    "Strength_Stock": (
                        round(strength_stock, 4)
                        if not np.isnan(strength_stock)
                        else None
                    ),
                    "Strength_Ref": round(strength_ref_cached, 4),
                    "RS": rs,
                    "Recomputed_RS": recomputed,
                    "Match": (
                        "OK"
                        if (
                            pd.notna(rs)
                            and pd.notna(recomputed)
                            and abs(rs - recomputed) < 0.1
                        )
                        else "MISMATCH"
                    )
                })

            except Exception as e:
                debug_records.append({
                    "Ticker": ticker,
                    "Days": 0,
                    "Start_Date": None,
                    "End_Date": None,
                    "Sufficient": "Error",
                    "3M_Return": None,
                    "6M_Return": None,
                    "9M_Return": None,
                    "12M_Return": None,
                    "Strength_Stock": None,
                    "Strength_Ref": round(strength_ref_cached, 4),
                    "RS": getattr(row, "RS", np.nan),
                    "Recomputed_RS": None,
                    "Match": f"Error: {e}"
                })

        for i, chunk in enumerate([
            debug_records[x:x + 3990]
            for x in range(0, len(debug_records), 3990)
        ]):
            pd.DataFrame(chunk).to_csv(
                os.path.join(debug_rs_dir, f"debug-rs-part{i + 1}.csv"),
                index=False
            )

        print(f"Full debug export done → {debug_rs_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate RS Rating for NSE India vs Benchmark"
    )

    parser.add_argument(
        "--arctic-db-path",
        default="tmp/arctic_db",
        help="Path to ArcticDB"
    )

    parser.add_argument(
        "--reference-ticker",
        default="^CRSLDX",
        help="Benchmark index"
    )

    parser.add_argument(
        "--output-dir",
        default="RS_Data",
        help="Output directory"
    )

    parser.add_argument(
        "--log-file",
        default="logs/calc.log",
        help="Log file"
    )

    parser.add_argument(
        "--metadata-file",
        default="data/ticker_price.json",
        help="Metadata JSON"
    )

    # parser.add_argument("--percentiles", default="98,89,69,49,29,9,1", help="Percentiles for RSRATING.csv")
    parser.add_argument("--percentiles", default="99,98,95,90,85,80,75,70,60,50,40,30,20,10,5,1", help="Comma-separated percentile thresholds")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable full debug export"
    )

    args = parser.parse_args()

    percentiles = [int(p) for p in args.percentiles.split(",")]
    percentiles = sorted({int(p.strip()) for p in args.percentiles.split(",") if p.strip()}, reverse=True)

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    main(
        args.arctic_db_path,
        args.reference_ticker,
        args.output_dir,
        args.log_file,
        args.metadata_file,
        percentiles,
        args.debug
    )
