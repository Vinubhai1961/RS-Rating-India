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
MAX_VOLUME_RETRIES = 2              # New: Dedicated volume retry attempts
VOLUME_RETRY_DELAY = (10, 18)
PRICE_THRESHOLD = 5.0

logging.basicConfig(
    level=logging.INFO,
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
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def parse_dvol(dvol):
    """Parse DVol string (e.g., '498.95K', '11.00M', '661') to integer."""
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
    except (ValueError, TypeError) as e:
        logging.debug(f"Failed to parse DVol '{dvol}': {e}")
        return None

def load_ticker_info():
    if not os.path.exists(TICKER_INFO_FILE):
        logging.error(f"{TICKER_INFO_FILE} not found!")
        return {}
    with open(TICKER_INFO_FILE, "r", encoding="utf-8") as f:
        try:
            return {item["ticker"]: {"info": {k: parse_dvol(v) if k == "DVol" else v for k, v in item["info"].items()}} for item in json.load(f)}
        except json.JSONDecodeError:
            logging.error("Invalid JSON in ticker_info.json")
            return {}

def partition_tickers(tickers, part_index, part_total):
    per_part = len(tickers) // part_total
    start = part_index * per_part
    end = start + per_part if part_index < part_total - 1 else len(tickers)
    return tickers[start:end]

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
    start_time = time.time()
    prices = []
    failure_reasons = {"no_price": 0, "below_threshold": 0, "missing_volume": 0, "error": 0}

    try:
        # Keep original logic intact
        yq = Ticker(batch, validate=False)

        try:
            hist = yq.history(period="1d")
        except Exception as e:
            logging.error(f"History fetch failed for batch: {e}")
            return 0, batch, []

        try:
            summary_details = yq.summary_detail
        except Exception as e:
            logging.warning(f"summary_detail fetch failed: {e}")
            summary_details = {}

        for symbol in batch:
            try:
                # -----------------------------
                # PRICE EXTRACTION (PATCHED)
                # -----------------------------
                price = None

                try:
                    if hasattr(hist, "index"):
                        symbols_in_hist = hist.index.get_level_values(0)

                        if symbol in symbols_in_hist:
                            symbol_hist = hist.loc[symbol]

                            if not symbol_hist.empty:
                                close_series = symbol_hist.get("close")

                                if close_series is not None and len(close_series) > 0:
                                    price = close_series.iloc[-1]

                except Exception as hist_err:
                    logging.debug(f"{symbol} history parse failed: {hist_err}")

                # Handle string prices safely
                if isinstance(price, str):
                    try:
                        price = float(price)
                    except:
                        price = None

                if price is None or not isinstance(price, (int, float)):
                    failure_reasons["no_price"] += 1
                    continue

                if price < PRICE_THRESHOLD:
                    failure_reasons["below_threshold"] += 1
                    continue

                # -----------------------------
                # SUMMARY DETAIL (PATCHED)
                # -----------------------------
                summary = {}

                if isinstance(summary_details, dict):
                    sym_summary = summary_details.get(symbol, {})

                    # Yahoo sometimes returns error strings
                    if isinstance(sym_summary, str):
                        logging.debug(f"Invalid summary for {symbol}: {sym_summary}")
                        sym_summary = {}

                    if isinstance(sym_summary, dict):
                        summary = sym_summary

                volume = summary.get("volume")
                avg_volume = summary.get("averageVolume")
                avg_volume_10days = summary.get("averageVolume10days")

                if avg_volume_10days is None or avg_volume_10days <= 0:
                    failure_reasons["missing_volume"] += 1

                fifty_two_week_low = summary.get("fiftyTwoWeekLow")
                fifty_two_week_high = summary.get("fiftyTwoWeekHigh")
                market_cap = summary.get("marketCap")

                info = ticker_info.get(symbol, {}).get("info", {})

                rvol = None
                if volume is not None and avg_volume_10days is not None and avg_volume_10days > 0:
                    rvol = f"{volume / avg_volume_10days:.2f}"

                prices.append({
                    "ticker": symbol,
                    "info": {
                        "Ticker Name": info.get("Ticker Name", "n/a"),
                        "Price": round(price, 2),
                        "DVol": format_volume(volume),
                        "RVol": rvol,
                        "Sector": info.get("Sector", "n/a").title(),
                        "Industry": info.get("Industry", "n/a").title(),
                        "type": "Stock",
                        "52WKL": round(fifty_two_week_low, 2) if fifty_two_week_low else None,
                        "52WKH": round(fifty_two_week_high, 2) if fifty_two_week_high else None,
                        "MCAP": format_market_cap(market_cap),
                        "AvgVol": format_volume(avg_volume),
                        "AvgVol10": format_volume(avg_volume_10days),
                        "Exchange": info.get("Exchange", "n/a"),
                        "FF": info.get("FF", None),
                        "1YR_Per": info.get("1YR_Per", None),
                        "DPChange": info.get("DPChange", None)
                    }
                })

            except Exception as e:
                failure_reasons["error"] += 1
                logging.debug(f"Failed to process {symbol}: {e}")

        failed_tickers = [s for s in batch if s not in [p["ticker"] for p in prices]]

        logging.info(
            f"Batch {'[Volume Retry]' if is_volume_retry else ''} stats: {failure_reasons}"
        )

        return len(prices), failed_tickers, prices

    except Exception as e:
        if "429" in str(e) or "curl" in str(e).lower():
            logging.warning(f"Rate limit hit in batch: {e}")
        else:
            logging.error(f"Unexpected batch error: {e}")

        return 0, batch, []
        
def main(part_index=None, part_total=None, verbose=False):
    ensure_dirs()
    setup_logging(verbose)

    start_time = time.time()
    start_time_str = datetime.now().strftime("%I:%M %p EDT on %A, %B %d, %Y")
    logging.info(f"Starting price build for part {part_index} at {start_time_str}")

    ticker_info = load_ticker_info()
    if not ticker_info:
        logging.error("No ticker_info.json found to process.")
        return

    qualified_tickers = list(ticker_info.keys())
    logging.info(f"Found {len(qualified_tickers)} tickers from ticker_info.json.")

    if part_index is not None and part_total is not None:
        part_tickers = partition_tickers(qualified_tickers, part_index, part_total)
        logging.info(f"Processing part {part_index}/{part_total} with {len(part_tickers)} tickers.")
    else:
        part_tickers = qualified_tickers

    batches = [part_tickers[i:i + BATCH_SIZE] for i in range(0, len(part_tickers), BATCH_SIZE)]
    all_prices = []
    all_failed = []
    volume_missing_tickers = []

    # Primary Processing
    for idx, batch in enumerate(tqdm(batches, desc="Processing Price Batches"), 1):
        updated, failed, prices = process_batch(batch, ticker_info)
        all_prices.extend(prices)
        all_failed.extend(failed)
        
        # Collect tickers missing AvgVol10 for retry
        for p in prices:
            if p["info"].get("AvgVol10") is None:
                volume_missing_tickers.append(p["ticker"])

        if idx < len(batches):
            delay = random.uniform(*BATCH_DELAY_RANGE)
            time.sleep(delay)

    # Dedicated Volume Retry Pass
    if volume_missing_tickers:
        volume_missing_tickers = list(set(volume_missing_tickers))
        logging.info(f"Starting dedicated volume retry for {len(volume_missing_tickers)} tickers...")
        vol_batches = [volume_missing_tickers[i:i + BATCH_SIZE] for i in range(0, len(volume_missing_tickers), BATCH_SIZE)]
        
        for vidx, vbatch in enumerate(tqdm(vol_batches, desc="Volume Retry"), 1):
            _, _, vol_prices = process_batch(vbatch, ticker_info, is_volume_retry=True)
            # Merge better data
            for vp in vol_prices:
                for i, ap in enumerate(all_prices):
                    if ap["ticker"] == vp["ticker"]:
                        all_prices[i] = vp
                        break
            time.sleep(random.uniform(*VOLUME_RETRY_DELAY))


    # Retry unresolved tickers
    unresolved_unique = sorted(set(all_failed))
    if unresolved_unique:
        logging.info(f"Retry sub-pass for {len(unresolved_unique)} unresolved tickers")
        retry_batches = [unresolved_unique[i:i + BATCH_SIZE] for i in range(0, len(unresolved_unique), BATCH_SIZE)]
        existing_tickers = {p["ticker"] for p in all_prices}
        for retry_batch in tqdm(retry_batches, desc="Retry Pass"):
            _, _, retry_prices = process_batch(retry_batch, ticker_info)
            for rp in retry_prices:
                if rp["ticker"] not in existing_tickers:
                    all_prices.append(rp)
                    existing_tickers.add(rp["ticker"])
            time.sleep(random.uniform(5, 10))

    # Save Results
    output_file = TICKER_PRICE_PART_FILE % part_index
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_prices, f, indent=2)
    
    missing_vol10 = sum(1 for x in all_prices if x["info"].get("AvgVol10") is None)
    logging.info(f"Saved {len(all_prices)} entries to {output_file}")
    logging.info(f"Final tickers with missing AvgVol10: {missing_vol10}")

    elapsed = time.time() - start_time
    logging.info("Price build completed. Elapsed: %.1fs", elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ticker_price.json from ticker_info.json.")
    parser.add_argument("--part-index", type=int, required=True)
    parser.add_argument("--part-total", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(part_index=args.part_index, part_total=args.part_total, verbose=args.verbose)
