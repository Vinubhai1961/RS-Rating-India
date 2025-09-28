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
UNRESOLVED_PRICE_TICKERS = os.path.join(OUTPUT_DIR, "unresolved_price_tickers.txt")
LOG_PATH = "logs/build_ticker_price.log"
BATCH_SIZE = 250
BATCH_DELAY_RANGE = (20, 30)  # Increased for two API calls
MAX_BATCH_RETRIES = 3
MAX_RETRY_TIMEOUT = 120
RETRY_SUBPASS = True
PRICE_THRESHOLD = 5.0  # Hardcoded minimum price threshold

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
        dvol = dvol.strip().upper()
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

def process_batch(batch, ticker_info):
    start_time = time.time()
    total_wait = 0
    for attempt in range(MAX_BATCH_RETRIES):
        try:
            prices = []
            failure_reasons = {"no_price": 0, "below_threshold": 0, "error": 0}
            # Pass tickers as they are from ticker_info.json
            yq = Ticker(batch)
            hist = yq.history(period="1d")
            summary_details = yq.summary_detail
            
            for symbol in batch:
                try:
                    # Extract price from history
                    price = None
                    if symbol in hist.index.get_level_values(0):
                        price = hist.loc[symbol]['close'].iloc[-1] if not hist.loc[symbol].empty else None
                    
                    # Validate price
                    if price is None or not isinstance(price, (int, float)):
                        logging.debug(f"Skipping {symbol}: No or invalid price data")
                        failure_reasons["no_price"] += 1
                        continue
                    if price < PRICE_THRESHOLD:
                        logging.debug(f"Skipping {symbol}: Price {price} below threshold {PRICE_THRESHOLD}")
                        failure_reasons["below_threshold"] += 1
                        continue
                    
                    # Extract summary details, set to None if missing or invalid
                    summary = summary_details.get(symbol, {}) if isinstance(summary_details, dict) else {}
                    none_fields = []
                    
                    volume = summary.get("volume")
                    if volume is None or not isinstance(volume, int) or volume < 0:
                        none_fields.append("DVol")
                        volume = None
                    
                    avg_volume = summary.get("averageVolume")
                    if avg_volume is None or not isinstance(avg_volume, int) or avg_volume < 0:
                        none_fields.append("AvgVol")
                        avg_volume = None
                    
                    avg_volume_10days = summary.get("averageVolume10days")
                    if avg_volume_10days is None or not isinstance(avg_volume_10days, int) or avg_volume_10days < 0:
                        none_fields.append("AvgVol10")
                        avg_volume_10days = None
                    
                    fifty_two_week_low = summary.get("fiftyTwoWeekLow")
                    if fifty_two_week_low is None or not isinstance(fifty_two_week_low, (int, float)) or fifty_two_week_low <= 0:
                        none_fields.append("52WKL")
                        fifty_two_week_low = None
                    
                    fifty_two_week_high = summary.get("fiftyTwoWeekHigh")
                    if fifty_two_week_high is None or not isinstance(fifty_two_week_high, (int, float)) or fifty_two_week_high <= 0:
                        none_fields.append("52WKH")
                        fifty_two_week_high = None
                    
                    market_cap = summary.get("marketCap")
                    if market_cap is None or not isinstance(market_cap, (int, float)) or market_cap < 0:
                        none_fields.append("MCAP")
                        market_cap = None
                    
                    if none_fields:
                        logging.debug(f"Ticker {symbol}: Setting {none_fields} to None due to missing or invalid summary data")
                    
                    # Get existing info from ticker_info.json
                    info = ticker_info.get(symbol, {}).get("info", {})
                    
                    # Calculate RVol as DVol / AvgVol10
                    rvol = None
                    if volume is not None and avg_volume_10days is not None and avg_volume_10days > 0:
                        rvol = f"{volume / avg_volume_10days:.2f}"
                    
                    # Combine data with preserved and updated fields
                    prices.append({
                        "ticker": symbol,
                        "info": {
                            "Ticker Name": info.get("Ticker Name", "n/a"),
                            "Price": round(price, 2),
                            "DVol": volume,
                            "RVol": rvol,
                            "sector": info.get("Sector", "n/a").lower(),
                            "industry": info.get("Industry", "n/a").lower(),
                            "type": "Stock",
                            "52WKL": round(fifty_two_week_low, 2) if fifty_two_week_low is not None else None,
                            "52WKH": round(fifty_two_week_high, 2) if fifty_two_week_high is not None else None,
                            "MCAP": round(market_cap, 2) if market_cap is not None else None,
                            "AvgVol": avg_volume,
                            "AvgVol10": avg_volume_10days,
                            "Exchange": info.get("Exchange", "n/a"),
                            "FF": info.get("FF", None),
                            "1YR_Per": info.get("1YR_Per", None),
                            "DPChange": info.get("DPChange", None)
                        }
                    })
                except Exception as e:
                    logging.debug(f"Failed to process {symbol}: {e}")
                    failure_reasons["error"] += 1
            
            failed_tickers = [s for s in batch if s not in [p["ticker"] for p in prices]]
            logging.info(f"Batch failure reasons: {failure_reasons}")
            return len(prices), failed_tickers, prices
        except Exception as e:
            if "429" in str(e) or "curl" in str(e).lower():
                wait = min((2 ** attempt) * random.uniform(5, 10), MAX_RETRY_TIMEOUT - total_wait)
                total_wait += wait
                if total_wait >= MAX_RETRY_TIMEOUT:
                    logging.warning(f"Max retry timeout reached for batch after {total_wait:.1f}s. Skipping.")
                    break
                logging.warning(f"Batch error (attempt {attempt+1}/{MAX_BATCH_RETRIES}): {e}. Retrying in {wait:.1f}s.")
                time.sleep(wait)
            else:
                logging.error(f"Unexpected error in batch: {e}. Aborting batch.")
                break
    return 0, batch, []

def main(part_index=None, part_total=None, verbose=False):
    start_time = time.time()
    ensure_dirs()
    setup_logging(verbose)

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

    for idx, batch in enumerate(tqdm(batches, desc="Processing Price Batches"), 1):
        updated, failed_tickers, prices = process_batch(batch, ticker_info)
        all_prices.extend(prices)
        all_failed.extend(failed_tickers)
        logging.info(f"Batch {idx}/{len(batches)} - Fetched data for {updated} tickers")
        if failed_tickers:
            logging.debug(f"Batch {idx}: Failed tickers: {failed_tickers}")
        if idx < len(batches):
            delay = random.uniform(*BATCH_DELAY_RANGE)
            logging.debug(f"Sleeping {delay:.1f}s before next batch...")
            time.sleep(delay)

    if RETRY_SUBPASS and all_failed:
        unresolved_unique = sorted(set(all_failed))
        logging.info(f"Retry sub-pass for {len(unresolved_unique)} unresolved tickers...")
        retry_batches = [unresolved_unique[i:i + BATCH_SIZE] for i in range(0, len(unresolved_unique), BATCH_SIZE)]
        for idx, batch in enumerate(tqdm(retry_batches, desc="Retry Price Batches"), 1):
            updated, failed_tickers, prices = process_batch(batch, ticker_info)
            all_prices.extend(prices)
            logging.info(f"Retry Batch {idx}/{len(retry_batches)} - Fetched data for {updated} tickers")
            if failed_tickers:
                logging.debug(f"Retry Batch {idx}: Failed tickers: {failed_tickers}")
            time.sleep(random.uniform(5, 10))

    unresolved_final = sorted(set(all_failed))
    with open(UNRESOLVED_PRICE_TICKERS, "w") as f:
        f.write("\n".join(unresolved_final))
    logging.info(f"Saved {len(unresolved_final)} unresolved tickers to {UNRESOLVED_PRICE_TICKERS}")

    output_file = TICKER_PRICE_PART_FILE % part_index
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_prices, f, indent=2)
    logging.info(f"Saved {len(all_prices)} entries to {output_file}")

    elapsed = time.time() - start_time
    logging.info("Price build completed. Elapsed: %.1fs", elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ticker_price.json from ticker_info.json.")
    parser.add_argument("--part-index", type=int, required=True)
    parser.add_argument("--part-total", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(part_index=args.part_index, part_total=args.part_total, verbose=args.verbose)
