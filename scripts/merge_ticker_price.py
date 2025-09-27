#!/usr/bin/env python3
import json
import os
import argparse
import logging
from glob import glob

OUTPUT_DIR = "data"
TICKER_PRICE_FILE = os.path.join(OUTPUT_DIR, "ticker_price.json")
UNRESOLVED_PRICE_TICKERS = os.path.join(OUTPUT_DIR, "unresolved_price_tickers.txt")
LOG_PATH = "logs/merge_ticker_price.log"

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

def merge_ticker_price(data_dir, part_total):
    ensure_dirs()
    logging.info(f"Merging ticker price data from {data_dir} with {part_total} partitions")

    # Merge ticker_price_part_X.json files
    all_prices = []
    for i in range(part_total):
        part_file = os.path.join(data_dir, f"ticker_price_part_{i}.json")
        if os.path.exists(part_file):
            try:
                with open(part_file, "r", encoding="utf-8") as f:
                    part_data = json.load(f)
                all_prices.extend(part_data)
                logging.info(f"Loaded {len(part_data)} entries from {part_file}")
            except Exception as e:
                logging.error(f"Failed to load {part_file}: {e}")
        else:
            logging.warning(f"Part file {part_file} not found")

    # Save merged ticker_price.json
    with open(TICKER_PRICE_FILE, "w", encoding="utf-8") as f:
        json.dump(all_prices, f, indent=2)
    logging.info(f"Saved {len(all_prices)} entries to {TICKER_PRICE_FILE}")

    # Merge unresolved_price_tickers_part_X.txt files
    all_unresolved = []
    for i in range(part_total):
        unresolved_file = os.path.join(data_dir, f"unresolved_price_tickers_part_{i}.txt")
        if os.path.exists(unresolved_file):
            try:
                with open(unresolved_file, "r") as f:
                    unresolved = f.read().splitlines()
                all_unresolved.extend(unresolved)
                logging.info(f"Loaded {len(unresolved)} unresolved tickers from {unresolved_file}")
            except Exception as e:
                logging.error(f"Failed to load {unresolved_file}: {e}")
        else:
            logging.warning(f"Unresolved file {unresolved_file} not found")

    # Save merged unresolved_price_tickers.txt
    unresolved_final = sorted(set(all_unresolved))
    with open(UNRESOLVED_PRICE_TICKERS, "w") as f:
        f.write("\n".join(unresolved_final))
    logging.info(f"Saved {len(unresolved_final)} unresolved tickers to {UNRESOLVED_PRICE_TICKERS}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ticker_price_part_X.json and unresolved_price_tickers_part_X.txt files.")
    parser.add_argument("data_dir", help="Directory containing partition files")
    parser.add_argument("--part-total", type=int, default=5, help="Total number of partitions")
    args = parser.parse_args()

    merge_ticker_price(args.data_dir, args.part_total)
