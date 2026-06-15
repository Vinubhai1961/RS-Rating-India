#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import os
import argparse
import logging
from yahooquery import Ticker
from tqdm import tqdm
from datetime import datetime
import time
import random

OUTPUT_DIR = "data"
TICKER_INFO_FILE = os.path.join(OUTPUT_DIR, "ticker_info.json")
TICKER_PRICE_PART_FILE = os.path.join(OUTPUT_DIR, "ticker_price_part_%d.json")
UNRESOLVED_PRICE_TICKERS = os.path.join(OUTPUT_DIR, "unresolved_price_tickers_part_%d.txt")
LOG_PATH = "logs/build_ticker_price.log"

BATCH_SIZE = 200
BATCH_DELAY_RANGE = (15, 25)
MAX_BATCH_RETRIES = 3
MAX_VOLUME_RETRIES = 2
VOLUME_RETRY_DELAY = (10, 18)
PRICE_THRESHOLD = 5.0

# Special tickers for detailed logging
SPECIAL_TICKERS = {"RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


def parse_dvol(dvol):
    if dvol is None or dvol == "":
        return None
    if isinstance(dvol, (int, float)):
        return int(dvol)
    try:
        dvol = str(dvol).strip().upper()
        if dvol.endswith("K"):
            return int(float(dvol[:-1]) * 1000)
        elif dvol.endswith("M"):
            return int(float(dvol[:-1]) * 1_000_000)
        else:
            return int(float(dvol))
    except Exception:
        logging.debug(f"Failed to parse DVol '{dvol}'")
        return None


def load_ticker_info():
    if not os.path.exists(TICKER_INFO_FILE):
        logging.error(f"{TICKER_INFO_FILE} not found!")
        return {}, []
    
    with open(TICKER_INFO_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, list):
                ticker_info = {}
                for item in data:
                    if isinstance(item, dict) and "ticker" in item:
                        ticker = item["ticker"]
                        info_data = item.get("info", item)
                        ticker_info[ticker] = {
                            "info": {k: parse_dvol(v) if k == "DVol" else v for k, v in info_data.items()}
                        }
            else:
                ticker_info = data  # dict format fallback

            qualified_tickers = sorted(ticker_info.keys())
            
            logging.info(f"Loaded {len(qualified_tickers)} NSE tickers")
            logging.info(f"First 5: {qualified_tickers[:5]}")
            logging.info(f"Last 5: {qualified_tickers[-5:]}")
            
            for special in SPECIAL_TICKERS:
                status = "✅ Found" if special in qualified_tickers else "❌ Missing"
                logging.info(f"{status} special ticker: {special}")
            
            return ticker_info, qualified_tickers
            
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in ticker_info.json: {e}")
            return {}, []


def partition_tickers(tickers, part_index, part_total):
    per_part = len(tickers) // part_total
    start = part_index * per_part
    end = start + per_part if part_index < part_total - 1 else len(tickers)
    return tickers[start:end]


def yahoo_symbol(symbol: str) -> str:
    """Ensure .NS suffix for Yahoo Finance"""
    if not symbol.endswith(".NS"):
        return f"{symbol}.NS"
    return symbol


def is_special(symbol):
    return symbol.replace(".NS", "") in SPECIAL_TICKERS


def format_volume(value):
    if value is None or value <= 0:
        return None
    return f"{value / 1000:.2f}K"


def format_market_cap(value):
    if value is None or value <= 0:
        return None
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    return f"{value / 1_000_000:.2f}M"


def process_batch(batch, ticker_info, is_volume_retry=False):
    for attempt in range(MAX_BATCH_RETRIES):
        try:
            prices = []
            failure_reasons = {"no_price": 0, "below_threshold": 0, "missing_volume": 0, "error": 0}

            yahoo_symbols = [yahoo_symbol(s) for s in batch]
            yq = Ticker(yahoo_symbols, validate=False)

            try:
                hist = yq.history(period="1d")
            except Exception as e:
                logging.error(f"History fetch failed: {e}")
                return 0, batch, []

            try:
                summary_details = yq.summary_detail
            except Exception:
                summary_details = {}

            for symbol in batch:
                yahoo_sym = yahoo_symbol(symbol)
                try:
                    price = None
                    source = "none"

                    # Primary price from history
                    if hasattr(hist, "index") and yahoo_sym in hist.index.get_level_values(0):
                        try:
                            df = hist.loc[yahoo_sym]
                            if not df.empty and 'close' in df.columns:
                                price = df['close'].iloc[-1]
                                source = "history"
                        except Exception:
                            pass

                    # Fallback
                    if price is None:
                        summary = summary_details.get(yahoo_sym, {}) if isinstance(summary_details, dict) else {}
                        for key in ["regularMarketPrice", "previousClose", "currentPrice", "price"]:
                            if key in summary and isinstance(summary[key], (int, float)):
                                price = summary[key]
                                source = f"summary.{key}"
                                break

                    if is_special(symbol):
                        logging.debug(f"{symbol} price from {source}: {price}")

                    if price is None or not isinstance(price, (int, float)):
                        failure_reasons["no_price"] += 1
                        continue

                    if price < PRICE_THRESHOLD and not is_special(symbol):
                        failure_reasons["below_threshold"] += 1
                        continue

                    # Get summary data
                    summary = summary_details.get(yahoo_sym, {}) if isinstance(summary_details, dict) else {}
                    if isinstance(summary, str):
                        summary = {}

                    info = ticker_info.get(symbol, {}).get("info", {})

                    volume = summary.get("volume")
                    avg_volume_10days = summary.get("averageVolume10days")

                    rvol = None
                    if volume and avg_volume_10days and avg_volume_10days > 0:
                        rvol = f"{volume / avg_volume_10days:.2f}"

                    if avg_volume_10days is None or avg_volume_10days <= 0:
                        failure_reasons["missing_volume"] += 1

                    prices.append({
                        "ticker": symbol,
                        "info": {
                            "Ticker Name": info.get("Ticker Name", "n/a"),
                            "Price": round(price, 2),
                            "DVol": format_volume(volume),
                            "RVol": rvol,
                            "Sector": str(info.get("Sector", "n/a")).title(),
                            "Industry": str(info.get("Industry", "n/a")).title(),
                            "type": "Stock",
                            "52WKL": round(summary.get("fiftyTwoWeekLow") or 0, 2),
                            "52WKH": round(summary.get("fiftyTwoWeekHigh") or 0, 2),
                            "MCAP": format_market_cap(summary.get("marketCap")),
                            "AvgVol": format_volume(summary.get("averageVolume")),
                            "AvgVol10": format_volume(avg_volume_10days),
                            "Exchange": info.get("Exchange", "NSE"),
                            "FF": info.get("FF"),
                            "1YR_Per": info.get("1YR_Per"),
                            "DPChange": info.get("DPChange"),
                            "Price_Source": source
                        }
                    })

                except Exception as e:
                    if is_special(symbol):
                        logging.error(f"{symbol} error: {e}")
                    failure_reasons["error"] += 1

            failed = [s for s in batch if s not in {p["ticker"] for p in prices}]
            logging.info(f"Batch {'[Vol Retry]' if is_volume_retry else ''} | Success: {len(prices)}/{len(batch)} | Fail: {failure_reasons}")
            return len(prices), failed, prices

        except Exception as e:
            logging.warning(f"Batch attempt {attempt+1} failed: {e}")
            time.sleep(random.uniform(8, 15))

    return 0, batch, []


def main(part_index=None, part_total=None, verbose=False):
    ensure_dirs()
    setup_logging(verbose)

    start_time = time.time()
    logging.info(f"Starting NSE price build - Part {part_index}/{part_total}")

    ticker_info, qualified_tickers = load_ticker_info()
    if not ticker_info:
        logging.error("Failed to load ticker_info.json")
        return

    if part_index is not None and part_total is not None:
        part_tickers = partition_tickers(qualified_tickers, part_index, part_total)
    else:
        part_tickers = qualified_tickers

    logging.info(f"Processing {len(part_tickers)} tickers in {len(part_tickers)//BATCH_SIZE + 1} batches")

    batches = [part_tickers[i:i + BATCH_SIZE] for i in range(0, len(part_tickers), BATCH_SIZE)]
    all_prices = []
    all_failed = []
    volume_missing = []

    for idx, batch in enumerate(tqdm(batches, desc="Price Batches"), 1):
        updated, failed, prices = process_batch(batch, ticker_info)
        all_prices.extend(prices)
        all_failed.extend(failed)

        for p in prices:
            if p["info"].get("AvgVol10") is None:
                volume_missing.append(p["ticker"])

        if idx < len(batches):
            time.sleep(random.uniform(*BATCH_DELAY_RANGE))

    # Volume Retry Pass
    if volume_missing:
        volume_missing = list(set(volume_missing))
        logging.info(f"Volume retry pass for {len(volume_missing)} tickers...")
        vol_batches = [volume_missing[i:i + BATCH_SIZE] for i in range(0, len(volume_missing), BATCH_SIZE)]
        for vbatch in tqdm(vol_batches, desc="Vol Retry"):
            _, _, vol_prices = process_batch(vbatch, ticker_info, is_volume_retry=True)
            for vp in vol_prices:
                for i, ap in enumerate(all_prices):
                    if ap["ticker"] == vp["ticker"]:
                        all_prices[i] = vp
                        break
            time.sleep(random.uniform(*VOLUME_RETRY_DELAY))

    # Save outputs
    output_file = TICKER_PRICE_PART_FILE % part_index
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_prices, f, indent=2)

    unresolved_file = UNRESOLVED_PRICE_TICKERS % part_index
    with open(unresolved_file, "w") as f:
        f.write("\n".join(sorted(set(all_failed))))

    missing_vol = sum(1 for x in all_prices if x["info"].get("AvgVol10") is None)
    logging.info(f"✅ Saved {len(all_prices)} tickers to {output_file}")
    logging.info(f"Missing AvgVol10: {missing_vol} | Unresolved: {len(all_failed)}")

    elapsed = time.time() - start_time
    logging.info(f"NSE Price build completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NSE ticker price data")
    parser.add_argument("--part-index", type=int, required=True)
    parser.add_argument("--part-total", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(part_index=args.part_index, part_total=args.part_total, verbose=args.verbose)
