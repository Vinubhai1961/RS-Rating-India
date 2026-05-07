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


# ============================================================
# ENHANCED TECHNICAL INDICATORS
# ============================================================
def calculate_smas_and_adr(df):
    """
    Institutional-grade technical indicator calculations.

    Calculates:
        - SMA50
        - SMA200
        - SMA10 Weekly
        - SMA30 Weekly
        - ADR(20)

    Handles:
        - duplicate timestamps
        - sparse datasets
        - NaN/inf values
        - invalid OHLC values
        - timezone inconsistencies
        - ArcticDB inconsistencies
    """

    sma50 = np.nan
    sma200 = np.nan
    sma10w = np.nan
    sma30w = np.nan
    adr = np.nan

    try:
        if df is None or len(df) == 0:
            return sma50, sma200, sma10w, sma30w, adr

        if 'close' not in df.columns:
            return sma50, sma200, sma10w, sma30w, adr

        df = df.copy()

        # ====================================================
        # INDEX CLEANUP
        # ====================================================
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')

        df = df[~df.index.isna()]

        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass

        df = df.sort_index()

        # Remove duplicate timestamps
        df = df.groupby(level=0).last()

        # ====================================================
        # CLEAN NUMERIC COLUMNS
        # ====================================================
        for col in ['open', 'high', 'low', 'close', 'volume', 'adjclose']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove inf values
        df = df.replace([np.inf, -np.inf], np.nan)

        closes = df['close'].dropna()

        if len(closes) == 0:
            return sma50, sma200, sma10w, sma30w, adr

        # ====================================================
        # DAILY SMA
        # ====================================================
        if len(closes) >= 50:
            sma50 = round(
                closes
                .rolling(window=50, min_periods=50)
                .mean()
                .iloc[-1],
                2
            )

        if len(closes) >= 200:
            sma200 = round(
                closes
                .rolling(window=200, min_periods=200)
                .mean()
                .iloc[-1],
                2
            )

        # ====================================================
        # WEEKLY SMA
        # ====================================================
        try:
            weekly_closes = (
                closes
                .resample('W-FRI')
                .last()
                .dropna()
            )

            if len(weekly_closes) >= 10:
                sma10w = round(
                    weekly_closes
                    .rolling(window=10, min_periods=10)
                    .mean()
                    .iloc[-1],
                    2
                )

            if len(weekly_closes) >= 30:
                sma30w = round(
                    weekly_closes
                    .rolling(window=30, min_periods=30)
                    .mean()
                    .iloc[-1],
                    2
                )

        except Exception:
            pass

        # ====================================================
        # ADR(20)
        # ====================================================
        if 'high' in df.columns and 'low' in df.columns:

            high = df['high']
            low = df['low']

            valid_mask = (
                high.notna() &
                low.notna() &
                (low > 0) &
                (high >= low)
            )

            if valid_mask.sum() >= 20:
                daily_range_pct = (
                    (high[valid_mask] / low[valid_mask]) - 1.0
                ) * 100.0

                adr_series = (
                    daily_range_pct
                    .rolling(window=20, min_periods=20)
                    .mean()
                    .dropna()
                )

                if len(adr_series) > 0:
                    adr = round(float(adr_series.iloc[-1]), 2)

        # ====================================================
        # FINAL SANITY CLEANUP
        # ====================================================
        values = {
            'sma50': sma50,
            'sma200': sma200,
            'sma10w': sma10w,
            'sma30w': sma30w,
            'adr': adr
        }

        for name, value in values.items():
            if pd.notna(value) and np.isinf(value):
                values[name] = np.nan

        sma50 = values['sma50']
        sma200 = values['sma200']
        sma10w = values['sma10w']
        sma30w = values['sma30w']
        adr = values['adr']

        return sma50, sma200, sma10w, sma30w, adr

    except Exception:
        return sma50, sma200, sma10w, sma30w, adr


# ============================================================
# LOGGING HELPERS
# ============================================================
def log_missing_rs(ticker: str, message: str, log_path: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ticker}] {message}\n")


# ============================================================
# ALIGNMENT HELPERS
# ============================================================
def align_series(closes, closes_ref):
    return pd.DataFrame({
        "stock": closes,
        "ref": closes_ref
    }).dropna().sort_index()


# ============================================================
# DEBUG HELPERS
# ============================================================
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


def debug_trend(ticker, rs_1m, rs_3m, rs_6m, log_path):
    if pd.notna(rs_1m) and pd.notna(rs_3m) and pd.notna(rs_6m):

        if rs_1m > rs_3m > rs_6m:
            log_missing_rs(ticker, "Trend: Accelerating 🚀", log_path)

        elif rs_1m < rs_3m < rs_6m:
            log_missing_rs(ticker, "Trend: Decelerating 📉", log_path)

        else:
            log_missing_rs(ticker, "Trend: Mixed", log_path)


# ============================================================
# RS CALCULATIONS
# ============================================================
def quarters_perf(closes: pd.Series, n: int) -> float:
    days = n * 63

    slice_len = min(len(closes), days + 1)
    available_data = closes[-slice_len:]

    if len(available_data) < 2:
        return 0.0 if len(available_data) == 1 else np.nan

    pct_change = available_data.pct_change(fill_method=None).dropna()

    if pct_change.empty:
        return np.nan

    return (pct_change + 1).cumprod().iloc[-1] - 1


# ============================================================
def strength(closes: pd.Series) -> float:
    perfs = [quarters_perf(closes, i) for i in range(1, 5)]

    valid_perfs = [p for p in perfs if not np.isnan(p)]

    if not valid_perfs:
        return np.nan

    weights = [0.4, 0.2, 0.2, 0.2][:len(valid_perfs)]

    total_weight = sum(weights)

    weights = [w / total_weight for w in weights]

    return sum(w * p for w, p in zip(weights, valid_perfs))


# ============================================================
def relative_strength(closes: pd.Series, closes_ref: pd.Series) -> float:
    rs_stock = strength(closes)
    rs_ref = strength(closes_ref)

    if np.isnan(rs_stock) or np.isnan(rs_ref):
        return np.nan

    rs = (1 + rs_stock) / (1 + rs_ref) * 100

    return round(rs, 2) if rs <= 700 else 700.0


# ============================================================
def short_relative_strength(closes, closes_ref, days):
    if len(closes) < days + 1:
        return np.nan

    if len(closes_ref) < days + 1:
        return np.nan

    price_old = closes.iloc[-days - 1]
    price_new = closes.iloc[-1]

    ref_old = closes_ref.iloc[-days - 1]
    ref_new = closes_ref.iloc[-1]

    if (
        price_new <= 0 or
        ref_new <= 0 or
        price_old <= 0 or
        ref_old <= 0
    ):
        return np.nan

    stock_ret = price_new / price_old - 1
    ref_ret = ref_new / ref_old - 1

    if ref_ret == 0:
        return np.nan if stock_ret <= 0 else 999.0

    rs = (1 + stock_ret) / (1 + ref_ret) * 100

    return round(rs, 2) if rs <= 700 else 700.0


# ============================================================
# ARCTICDB
# ============================================================
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


# ============================================================
# TRADINGVIEW CSV
# ============================================================
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
        for cal_name in ['NSE', 'XBOM']:
            try:
                cal = get_calendar(cal_name)

                sched = cal.schedule(
                    start_date=latest_date - timedelta(days=20),
                    end_date=latest_date + timedelta(days=2)
                )

                valid_dates = [
                    d.date() for d in sched.index
                    if d.date() <= latest_date
                ]

                dates = [
                    d.strftime('%Y%m%dT')
                    for d in valid_dates[-5:]
                ]

                if len(dates) >= 5:
                    break

            except Exception as e:
                logging.warning(f"{cal_name} calendar failed: {e}")

    if len(dates) < 5:
        dates = [
            (latest_date - timedelta(days=i)).strftime('%Y%m%dT')
            for i in range(4, -1, -1)
        ]

    valid_rs = (
        df_stocks['RS']
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

    path = os.path.join(output_dir, 'RSRATING.csv')

    with open(path, 'w') as f:
        f.write(''.join(lines))

    return ''.join(lines)


# ============================================================
# MAIN
# ============================================================
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
        format='%(asctime)s - %(message)s'
    )

    logging.info(
        'Starting NSE India RS calculation with Technical Indicators'
    )

    debug_rs_dir = os.path.join(
        os.path.dirname(log_file),
        'debug_rs'
    )

    os.makedirs(debug_rs_dir, exist_ok=True)

    missing_rs_log = os.path.join(
        debug_rs_dir,
        'Missing_RS.log'
    )

    open(missing_rs_log, 'w').close()

    # ========================================================
    # LOAD DATABASE
    # ========================================================
    result = load_arctic_db(arctic_db_path)

    if not result:
        logging.error('Failed to load ArcticDB. Exiting.')
        sys.exit(1)

    lib, tickers = result

    if reference_ticker not in tickers:
        logging.error(f'Reference ticker {reference_ticker} not found')
        sys.exit(1)

    # ========================================================
    # REFERENCE DATA
    # ========================================================
    ref_data = lib.read(reference_ticker).data

    ref_closes = pd.Series(
        pd.to_numeric(ref_data['close'], errors='coerce').values,
        index=pd.to_datetime(ref_data['datetime'], unit='s')
    ).sort_index()

    ref_closes = ref_closes.groupby(level=0).last()

    ref_closes = (
        ref_closes
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(ref_closes) < 20:
        logging.error('Not enough reference ticker data')
        sys.exit(1)

    # ========================================================
    # METADATA
    # ========================================================
    metadata_df = pd.DataFrame()

    if metadata_file and os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)

            records = []

            if isinstance(data, list):
                for item in data:
                    info = item.get('info', {})

                    records.append({
                        'Ticker': item.get('ticker'),
                        'Price': info.get('Price'),
                        'DVol': info.get('DVol', ''),
                        'Sector': info.get('Sector', ''),
                        'Industry': info.get('Industry', ''),
                        'AvgVol': info.get('AvgVol', ''),
                        'AvgVol10': info.get('AvgVol10', ''),
                        '52WKH': info.get('52WKH'),
                        '52WKL': info.get('52WKL'),
                        'MCAP': info.get('MCAP'),
                        'Type': info.get('type', 'Stock')
                    })

            metadata_df = pd.DataFrame([
                r for r in records
                if r['Ticker']
            ])

        except Exception as e:
            logging.error(f'Invalid metadata file: {e}')

    # ========================================================
    print(f'Processing {len(tickers)-1:,} stocks...')

    rs_results = []
    valid_rs_count = 0

    # ========================================================
    # TICKER LOOP
    # ========================================================
    for ticker in tqdm(tickers, desc='Calculating RS + Indicators'):

        if ticker == reference_ticker:
            continue

        try:
            data_obj = lib.read(ticker)
            df_data = data_obj.data

            # =================================================
            # DATA VALIDATION
            # =================================================
            if df_data is None or len(df_data) == 0:
                log_missing_rs(
                    ticker,
                    'EMPTY DATAFRAME',
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
                    np.nan
                ))

                continue

            # =================================================
            # CLOSE SERIES
            # =================================================
            closes = pd.Series(
                pd.to_numeric(df_data['close'], errors='coerce').values,
                index=pd.to_datetime(df_data['datetime'], unit='s')
            ).sort_index()

            closes = closes.groupby(level=0).last()

            closes = (
                closes
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(closes) < 2:
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
                    np.nan
                ))
                continue

            # =================================================
            # PRICE DATAFRAME
            # =================================================
            price_df = pd.DataFrame(
                index=pd.to_datetime(df_data['datetime'], unit='s')
            )

            for col in [
                'open',
                'high',
                'low',
                'close',
                'volume',
                'adjclose'
            ]:
                if col in df_data.columns:
                    price_df[col] = pd.to_numeric(
                        df_data[col],
                        errors='coerce'
                    )

            price_df = price_df[~price_df.index.isna()]

            price_df = price_df.groupby(level=0).last()

            price_df = price_df.replace(
                [np.inf, -np.inf],
                np.nan
            )

            price_df = price_df.sort_index()

            price_df = price_df.dropna(how='all')

            # =================================================
            # INDICATORS
            # =================================================
            sma50, sma200, sma10w, sma30w, adr = (
                calculate_smas_and_adr(price_df)
            )

            log_missing_rs(
                ticker,
                f'INDICATORS → '
                f'Rows={len(price_df)} | '
                f'SMA50={sma50} | '
                f'SMA200={sma200} | '
                f'SMA10W={sma10w} | '
                f'SMA30W={sma30w} | '
                f'ADR={adr}',
                missing_rs_log
            )

            # =================================================
            # RS
            # =================================================
            rs = relative_strength(closes, ref_closes)

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
                '1M',
                missing_rs_log,
                reference_ticker
            )

            s3, r3 = debug_returns(
                ticker,
                df_aligned,
                63,
                '3M',
                missing_rs_log,
                reference_ticker
            )

            s6, r6 = debug_returns(
                ticker,
                df_aligned,
                126,
                '6M',
                missing_rs_log,
                reference_ticker
            )

            rs_1m = short_relative_strength(closes, ref_closes, 21)
            rs_3m = short_relative_strength(closes, ref_closes, 63)
            rs_6m = short_relative_strength(closes, ref_closes, 126)

            validate_rs(
                ticker,
                rs_1m,
                s1,
                r1,
                '1M',
                missing_rs_log
            )

            validate_rs(
                ticker,
                rs_3m,
                s3,
                r3,
                '3M',
                missing_rs_log
            )

            validate_rs(
                ticker,
                rs_6m,
                s6,
                r6,
                '6M',
                missing_rs_log
            )

            debug_trend(
                ticker,
                rs_1m,
                rs_3m,
                rs_6m,
                missing_rs_log
            )

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
                adr
            ))

            if not np.isnan(rs):
                valid_rs_count += 1

        except Exception as e:
            log_missing_rs(
                ticker,
                f'EXCEPTION: {e}',
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
                np.nan
            ))

    # ========================================================
    # OUTPUT
    # ========================================================
    df_stocks = pd.DataFrame(
        rs_results,
        columns=[
            'Ticker',
            'RS',
            '1M_RS',
            '3M_RS',
            '6M_RS',
            'SMA50',
            'SMA200',
            'SMA10_Weekly',
            'SMA30_Weekly',
            'ADR'
        ]
    )

    if not metadata_df.empty:
        df_stocks = df_stocks.merge(
            metadata_df,
            on='Ticker',
            how='left'
        )

    for col in ['RS', '1M_RS', '3M_RS', '6M_RS']:
        valid = df_stocks[col].dropna()

        if not valid.empty:
            df_stocks.loc[
                valid.index,
                f'{col} Percentile'
            ] = (
                valid.rank(pct=True, method='min') * 99
            ).astype(int)

    df_stocks = (
        df_stocks
        .sort_values('RS', ascending=False, na_position='last')
        .reset_index(drop=True)
    )

    df_stocks['Rank'] = df_stocks.index + 1

    df_stocks['IPO'] = 'No'

    # ========================================================
    # STAGE HELPERS
    # ========================================================
    if 'Price' in df_stocks.columns:

        df_stocks['Price'] = pd.to_numeric(
            df_stocks['Price'],
            errors='coerce'
        )

        df_stocks['Above_SMA50'] = np.where(
            df_stocks['Price'] > df_stocks['SMA50'],
            'Yes',
            'No'
        )

        df_stocks['Above_SMA200'] = np.where(
            df_stocks['Price'] > df_stocks['SMA200'],
            'Yes',
            'No'
        )

        df_stocks['SMA50_GT_SMA200'] = np.where(
            df_stocks['SMA50'] > df_stocks['SMA200'],
            'Yes',
            'No'
        )

    if 'Type' in df_stocks.columns:
        df_stocks.loc[
            df_stocks['Type'] == 'ETF',
            ['Sector', 'Industry']
        ] = 'ETF'

    os.makedirs(output_dir, exist_ok=True)

    output_columns = [
        'Rank',
        'Ticker',
        'Price',
        'DVol',
        'Sector',
        'Industry',
        'RS Percentile',
        '1M_RS Percentile',
        '3M_RS Percentile',
        '6M_RS Percentile',
        'SMA50',
        'SMA200',
        'SMA10_Weekly',
        'SMA30_Weekly',
        'ADR',
        'Above_SMA50',
        'Above_SMA200',
        'SMA50_GT_SMA200',
        'AvgVol',
        'AvgVol10',
        '52WKH',
        '52WKL',
        'MCAP',
        'IPO'
    ]

    available_cols = [
        col for col in output_columns
        if col in df_stocks.columns
    ]

    df_stocks[available_cols].to_csv(
        os.path.join(output_dir, 'rs_stocks.csv'),
        index=False,
        na_rep=''
    )

    # ========================================================
    # INDUSTRY TABLE
    # ========================================================
    df_industries = df_stocks.groupby('Industry').agg({
        'RS Percentile': 'mean',
        '1M_RS Percentile': 'mean',
        '3M_RS Percentile': 'mean',
        '6M_RS Percentile': 'mean',
        'Sector': 'first',
        'Ticker': lambda x: ','.join(
            df_stocks[
                df_stocks['Ticker'].isin(x)
            ]
            .sort_values('RS', ascending=False)['Ticker']
        )
    }).reset_index()

    for col in [
        'RS Percentile',
        '1M_RS Percentile',
        '3M_RS Percentile',
        '6M_RS Percentile'
    ]:
        df_industries[col] = (
            df_industries[col]
            .fillna(0)
            .round()
            .astype(int)
        )

    df_industries = (
        df_industries
        .sort_values('RS Percentile', ascending=False)
        .reset_index(drop=True)
    )

    df_industries['Rank'] = df_industries.index + 1

    df_industries.rename(columns={
        'RS Percentile': 'RS',
        '1M_RS Percentile': '1M_RS',
        '3M_RS Percentile': '3M_RS',
        '6M_RS Percentile': '6M_RS'
    }, inplace=True)

    df_industries[
        [
            'Rank',
            'Industry',
            'Sector',
            'RS',
            '1M_RS',
            '3M_RS',
            '6M_RS',
            'Ticker'
        ]
    ].to_csv(
        os.path.join(output_dir, 'rs_industries.csv'),
        index=False
    )

    # ========================================================
    # TRADINGVIEW CSV
    # ========================================================
    generate_tradingview_csv(
        df_stocks,
        output_dir,
        ref_data,
        percentiles
    )

    # ========================================================
    print('\n✅ ENHANCED RS CALCULATION COMPLETE!')
    print(f'Valid RS: {valid_rs_count:,} / {len(df_stocks):,}')
    print(f'Output → {output_dir}/')


# ============================================================
# ENTRY
# ============================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Enhanced RS Rating Calculation Engine'
    )

    parser.add_argument(
        '--arctic-db-path',
        default='tmp/arctic_db',
        help='Path to ArcticDB'
    )

    parser.add_argument(
        '--reference-ticker',
        default='^CRSLDX',
        help='Benchmark index'
    )

    parser.add_argument(
        '--output-dir',
        default='RS_Data',
        help='Output directory'
    )

    parser.add_argument(
        '--log-file',
        default='logs/calc.log',
        help='Log file'
    )

    parser.add_argument(
        '--metadata-file',
        default='data/ticker_price.json',
        help='Metadata JSON'
    )

    parser.add_argument(
        '--percentiles',
        default='98,89,69,49,29,9,1',
        help='Percentiles for RSRATING.csv'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()

    percentiles = [
        int(p)
        for p in args.percentiles.split(',')
    ]

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
