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
BATCH_SIZE = 250
BATCH_DELAY_RANGE = (20, 30)  # Increased for two API calls
MAX_BATCH_RETRIES = 3
MAX_RETRY_TIMEOUT = 120
RETRY_SUBPASS = True
PRICE_THRESHOLD = 5.0  # Hardcoded minimum price threshold
MAX_VOLUME_RETRY = 2          # Extra retries if volume is missing
VOLUME_RETRY_DELAY = (8, 15)

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

def format_volume(value):
    """Format volume as a string with 'K' suffix for thousands."""
    if value is None:
        return None
    return f"{value / 1000:.2f}K"

def format_market_cap(value):
    """Format market cap as a string with 'M' for millions or 'B' for billions."""
    if value is None:
        return None
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    return f"{value / 1_000_000:.2f}M"

def process_batch(batch, ticker_info):
    start_time = time.time()
    total_wait = 0
    for attempt in range(MAX_BATCH_RETRIES):
        try:
            prices = []
            failure_reasons = {"no_price": 0, "below_threshold": 0, "missing_volume": 0, "error": 0}
            
            yq = Ticker(batch)
            hist = yq.history(period="1d")
            summary_details = yq.summary_detail
            
            for symbol in batch:
                try:
                    # Price extraction
                    price = None
                    if symbol in hist.index.get_level_values(0):
                        price = hist.loc[symbol]['close'].iloc[-1] if not hist.loc[symbol].empty else None
                    
                    if price is None or not isinstance(price, (int, float)):
                        failure_reasons["no_price"] += 1
                        continue
                    if price < PRICE_THRESHOLD:
                        failure_reasons["below_threshold"] += 1
                        continue

                    summary = summary_details.get(symbol, {}) if isinstance(summary_details, dict) else {}
                    
                    # Volume fields
                    volume = summary.get("volume")
                    avg_volume = summary.get("averageVolume")
                    avg_volume_10days = summary.get("averageVolume10days")

                    # Track missing volume
                    missing_vol = []
                    if avg_volume is None or not isinstance(avg_volume, (int, float)) or avg_volume <= 0:
                        missing_vol.append("AvgVol")
                    if avg_volume_10days is None or not isinstance(avg_volume_10days, (int, float)) or avg_volume_10days <= 0:
                        missing_vol.append("AvgVol10")

                    if missing_vol:
                        failure_reasons["missing_volume"] += 1

                    # === NEW: Retry individual ticker if volume is missing (in next pass) ===
                    if missing_vol and attempt == 0:  # Only on first attempt
                        logging.debug(f"{symbol}: Missing volume data ({missing_vol}) - will retry individually later")

                    # Format functions (same as before)
                    def format_volume(v):
                        return f"{v/1000:.2f}K" if v and v > 0 else None

                    def format_market_cap(v):
                        if not v: return None
                        if v >= 1_000_000_000:
                            return f"{v/1_000_000_000:.2f}B"
                        return f"{v/1_000_000:.2f}M"

                    info = ticker_info.get(symbol, {}).get("info", {})

                    rvol = None
                    if volume and avg_volume_10days and avg_volume_10days > 0:
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
                            "52WKL": round(summary.get("fiftyTwoWeekLow"), 2) if summary.get("fiftyTwoWeekLow") else None,
                            "52WKH": round(summary.get("fiftyTwoWeekHigh"), 2) if summary.get("fiftyTwoWeekHigh") else None,
                            "MCAP": format_market_cap(summary.get("marketCap")),
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
                    logging.debug(f"Failed {symbol}: {e}")

            failed_tickers = [s for s in batch if s not in [p["ticker"] for p in prices]]
            logging.info(f"Batch failure reasons: {failure_reasons}")
            return len(prices), failed_tickers, prices

        except Exception as e:
            # Existing rate limit retry logic (unchanged)
            if "429" in str(e) or "curl" in str(e).lower():
                wait = min((2 ** attempt) * random.uniform(5, 10), MAX_RETRY_TIMEOUT - total_wait)
                total_wait += wait
                if total_wait >= MAX_RETRY_TIMEOUT:
                    break
                logging.warning(f"Batch error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s.")
                time.sleep(wait)
            else:
                logging.error(f"Unexpected error: {e}")
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
    with open(UNRESOLVED_PRICE_TICKERS % part_index, "w") as f:
        f.write("\n".join(unresolved_final))
    logging.info(f"Saved {len(unresolved_final)} unresolved tickers to {UNRESOLVED_PRICE_TICKERS % part_index}")

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
