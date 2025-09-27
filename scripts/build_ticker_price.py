#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import os
import pandas as pd
import logging
from yahooquery import Ticker
from tqdm import tqdm
from datetime import datetime
import time
import random

OUTPUT_DIR = "data"
TICKER_PRICE_FILE = os.path.join(OUTPUT_DIR, "ticker_price.json")
UNRESOLVED_PRICE_TICKERS = os.path.join(OUTPUT_DIR, "unresolved_price_tickers.txt")
LOG_PATH = "logs/build_ticker_price.log"
BATCH_SIZE = 250
BATCH_DELAY_RANGE = (20, 30)
MAX_BATCH_RETRIES = 3
MAX_RETRY_TIMEOUT = 120
RETRY_SUBPASS = True
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

def load_ticker_info():
    input_path = os.path.join(OUTPUT_DIR, "test.csv")
    if not os.path.exists(input_path):
        logging.error(f"{input_path} not found!")
        return None
    df = pd.read_csv(input_path)
    return df

def yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")

def process_batch(batch, ticker_df):
    start_time = time.time()
    total_wait = 0
    for attempt in range(MAX_BATCH_RETRIES):
        try:
            prices = []
            failure_reasons = {"no_price": 0, "below_threshold": 0, "error": 0}
            yahoo_symbols = [yahoo_symbol(symbol) for symbol in batch]
            yq = Ticker(yahoo_symbols)
            hist = yq.history(period="1d")
            summary_details = yq.summary_detail

            for symbol in batch:
                yahoo_sym = yahoo_symbol(symbol)
                try:
                    # Extract price from history
                    price = None
                    if yahoo_sym in hist.index.get_level_values(0):
                        price = hist.loc[yahoo_sym]['close'].iloc[-1] if not hist.loc[yahoo_sym].empty else None

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
                    summary = summary_details.get(yahoo_sym, {}) if isinstance(summary_details, dict) else {}
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

                    # Get metadata from test.csv
                    ticker_row = ticker_df[ticker_df['Ticker'] == symbol].iloc[0]
                    prices.append({
                        "ticker": symbol,
                        "info": {
                            "Price": round(price, 2),
                            "industry": ticker_row['Industry'] if pd.notna(ticker_row['Industry']) else "n/a",
                            "sector": ticker_row['Sector'] if pd.notna(ticker_row['Sector']) else "n/a",
                            "type": "Unknown",  # Not in test.csv; set to Unknown
                            "DVol": volume,
                            "AvgVol": avg_volume,
                            "AvgVol10": avg_volume_10days,
                            "52WKL": round(fifty_two_week_low, 2) if fifty_two_week_low is not None else None,
                            "52WKH": round(fifty_two_week_high, 2) if fifty_two_week_high is not None else None,
                            "MCAP": round(market_cap, 2) if market_cap is not None else None,
                            "RVol": round(ticker_row['RVol'], 2) if pd.notna(ticker_row['RVol']) else None,
                            "FF": round(ticker_row['FF'], 2) if pd.notna(ticker_row['FF']) else None,
                            "1YR_Per": round(ticker_row['1YR_Per'], 2) if pd.notna(ticker_row['1YR_Per']) else None
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

def main(verbose=False):
    start_time = time.time()
    ensure_dirs()
    setup_logging(verbose)

    start_time_str = datetime.now().strftime("%I:%M %p EDT on %A, %B %d, %Y")
    logging.info(f"Starting price build at {start_time_str}")

    ticker_df = load_ticker_info()
    if ticker_df is None:
        logging.error("No test.csv found to process.")
        return

    tickers = ticker_df['Ticker'].tolist()
    logging.info(f"Found {len(tickers)} tickers from test.csv.")

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    all_prices = []
    all_failed = []

    for idx, batch in enumerate(tqdm(batches, desc="Processing Price Batches"), 1):
        updated, failed_tickers, prices = process_batch(batch, ticker_df)
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
            updated, failed_tickers, prices = process_batch(batch, ticker_df)
            all_prices.extend(prices)
            logging.info(f"Retry Batch {idx}/{len(retry_batches)} - Fetched data for {updated} tickers")
            if failed_tickers:
                logging.debug(f"Retry Batch {idx}: Failed tickers: {failed_tickers}")
            time.sleep(random.uniform(5, 10))

    unresolved_final = sorted(set(all_failed))
    with open(UNRESOLVED_PRICE_TICKERS, "w") as f:
        f.write("\n".join(unresolved_final))
    logging.info(f"Saved {len(unresolved_final)} unresolved tickers to {UNRESOLVED_PRICE_TICKERS}")

    with open(TICKER_PRICE_FILE, "w", encoding="utf-8") as f:
        json.dump(all_prices, f, indent=2)
    logging.info(f"Saved {len(all_prices)} entries to {TICKER_PRICE_FILE}")

    elapsed = time.time() - start_time
    logging.info("Price build completed. Elapsed: %.1fs", elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ticker_price.json from test.csv.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(verbose=args.verbose)
