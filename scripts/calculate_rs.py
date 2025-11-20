#!/usr/bin/env python3
import os
import time
import json
import argparse
import logging
from tqdm import tqdm
from yahooquery import Ticker
import pandas as pd
import arcticdb as adb

def fetch_historical_data(tickers, arctic, log_file):
    max_retries = 3
    batch_size = 200
    failed_tickers = []
    skipped_tickers = []
    success_tickers = []

    lib = arctic.get_library("prices", create_if_missing=True)
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    logging.info(f"Starting fetch for {len(tickers)} tickers in {total_batches} batches of {batch_size}")

    for i in tqdm(range(0, len(tickers), batch_size), total=total_batches, desc="Fetching batches"):
        batch = tickers[i:i + batch_size]
        batch_success = 0
        batch_skipped = 0
        batch_start_time = time.time()

        for attempt in range(max_retries):
            try:
                data = Ticker(batch).history(period="2y")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Batch {i//batch_size + 1} failed after {max_retries} attempts: {str(e)}")
                    failed_tickers.extend([(t, str(e)) for t in batch])
                    data = None
                else:
                    logging.warning(f"Retrying batch {i//batch_size + 1} (attempt {attempt+2}/{max_retries}) after error: {str(e)}")
                    time.sleep(2 ** attempt)  # exponential backoff

        if data is None:
            continue

        for ticker in batch:
            try:
                if ticker not in data.index.get_level_values(0):
                    skipped_tickers.append((ticker, "No data returned"))
                    batch_skipped += 1
                    continue

                df = data.loc[ticker].reset_index()

                if df.empty:
                    skipped_tickers.append((ticker, "Empty DataFrame"))
                    batch_skipped += 1
                    continue

                df = df.rename(columns={"date": "datetime"})
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True).astype(int) // 10**9

                lib.write(ticker, df)
                success_tickers.append(ticker)
                batch_success += 1

            except Exception as e:
                failed_tickers.append((ticker, str(e)))

        batch_time = time.time() - batch_start_time
        logging.info(f"✅ Batch {i//batch_size + 1}/{total_batches} | Success: {batch_success} | Skipped: {batch_skipped} | Time: {batch_time:.2f}s")

    # Final summary
    with open(log_file, "a") as f:
        f.write("\n=== FINAL SUMMARY ===\n")
        f.write(f"Successful: {len(success_tickers)}\n")
        f.write(f"Skipped: {len(skipped_tickers)}\n")
        f.write(f"Failed: {len(failed_tickers)}\n")
        if skipped_tickers:
            f.write("\n--- Skipped Tickers ---\n")
            for t, r in skipped_tickers[:200]:  # limit log spam
                f.write(f"{t}: {r}\n")
        if failed_tickers:
            f.write("\n--- Failed Tickers ---\n")
            for t, e in failed_tickers[:200]:
                f.write(f"{t}: {e}\n")

    print(f"\n🎉 Fetch complete! → {len(success_tickers)} stored | {len(skipped_tickers)} skipped | {len(failed_tickers)} failed")


def load_ticker_list(file_path, partition=None, total_partitions=None):
    with open(file_path, "r") as f:
        data = json.load(f)

    # Your file has 67,517 lines → it's a list of objects
    if isinstance(data, list):
        tickers = [item["ticker"] for item in data if isinstance(item, dict) and "ticker" in item]
    else:
        raise ValueError("Expected ticker_price.json to be a list of ticker objects")

    total_tickers = len(tickers)
    logging.info(f"Loaded {total_tickers:,} tickers from ticker_price.json")

    # ABSOLUTELY CRITICAL FOR 67K+ TICKERS:
    # Force NIFTY 50 to be fetched first (critical for full-market RS)
    BENCHMARK = "^NSEI"
    if BENCHMARK in tickers:
        tickers.remove(BENCHMARK)
        tickers.insert(0, BENCHMARK)
        logging.info(f"Prioritized benchmark {BENCHMARK} → always in partition 0")

    if partition is not None and total_partitions:
        chunk_size = total_tickers // total_partitions
        start = partition * chunk_size
        # Last partition gets the remainder
        end = None if partition == total_partitions - 1 else start + chunk_size
        partition_tickers = tickers[start:end]
        logging.info(f"Partition {partition}/{total_partitions} → {len(partition_tickers):,} tickers (indices {start} to {end or 'end'})")
        return partition_tickers

    return tickers


def main():
    parser = argparse.ArgumentParser(description="Fetch 2Y data for ~67,500 Indian tickers → ArcticDB")
    parser.add_argument("input_file", help="Path to data/ticker_price.json")
    parser.add_argument("--log-file", default="logs/fetch_log.log", help="Log file")
    parser.add_argument("--arctic-db-path", default="tmp/arctic_db", help="ArcticDB path")
    parser.add_argument("--partition", type=int, default=None)
    parser.add_argument("--total-partitions", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(args.arctic_db_path, exist_ok=True)
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format="%(asctime)s | %(message)s")

    tickers = load_ticker_list(args.input_file, args.partition, args.total_partitions)
    arctic = adb.Arctic(f"lmdb://{args.arctic_db_path}")
    fetch_historical_data(tickers, arctic, args.log_file)


if __name__ == "__main__":
    main()
