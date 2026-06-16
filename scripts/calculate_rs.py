#!/usr/bin/env python3
import os
import time
import json
import argparse
import logging
from tqdm import tqdm
from yahooquery import Ticker
import pandas as pd
import numpy as np
import arcticdb as adb


# ================== SPECIAL TICKERS (Force Save Short History) ==================
# Keep empty for NSE by default. Add IPO/SME tickers here only when you intentionally
# want to save symbols with fewer than MIN_VALID_ROWS rows.
#
# Example:
SPECIAL_TICKERS = {"AZAD.NS", "PREMEXPLN.NS"}


# NSE should stay stricter than USA for RS quality.
MIN_VALID_ROWS = 5


def fetch_historical_data(tickers, arctic, log_file):
    max_retries = 3
    batch_size = 200

    failed_tickers = []
    skipped_tickers = []
    success_tickers = []

    lib = arctic.get_library("prices", create_if_missing=True)
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    logging.info(
        f"Starting fetch for {len(tickers)} tickers "
        f"in {total_batches} batches of {batch_size}"
    )
    logging.info(f"Special tickers bypassing short-history rule: {SPECIAL_TICKERS}")
    logging.info(f"Minimum valid rows required: {MIN_VALID_ROWS}")

    for i in tqdm(
        range(0, len(tickers), batch_size),
        total=total_batches,
        desc="Fetching batches"
    ):
        batch = tickers[i:i + batch_size]
        batch_no = i // batch_size + 1

        batch_success = 0
        batch_skipped = 0
        batch_failed = 0
        batch_skipped_list = []
        batch_failed_list = []
        batch_start_time = time.time()

        # ================= FETCH WITH RETRY =================
        data = None

        for attempt in range(max_retries):
            try:
                data = Ticker(batch).history(period="2y")
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(
                        f"Batch {batch_no}/{total_batches} failed "
                        f"after {max_retries} attempts: {str(e)}"
                    )

                    failed_tickers.extend(
                        [(ticker, str(e)) for ticker in batch]
                    )

                    batch_failed = len(batch)
                    batch_failed_list.extend(batch)
                    data = None

                else:
                    logging.warning(
                        f"Retrying batch {batch_no}/{total_batches} "
                        f"(attempt {attempt + 2}/{max_retries}) "
                        f"after error: {str(e)}"
                    )
                    time.sleep(2)

        if data is None:
            batch_time = time.time() - batch_start_time
            logging.info(
                f"❌ Batch {batch_no}/{total_batches} failed completely "
                f"in {batch_time:.2f}s | "
                f"Success: {batch_success} | "
                f"Skipped: {batch_skipped} | "
                f"Failed: {batch_failed}"
            )
            continue

        # ================= PROCESS EACH TICKER =================
        for ticker in batch:
            try:
                if ticker not in data.index.get_level_values(0):
                    reason = "No data returned from Yahoo"
                    skipped_tickers.append((ticker, reason))
                    batch_skipped_list.append(ticker)
                    batch_skipped += 1
                    continue

                df = data.loc[ticker].reset_index()

                if df.empty:
                    reason = "Empty DataFrame"
                    skipped_tickers.append((ticker, reason))
                    batch_skipped_list.append(ticker)
                    batch_skipped += 1
                    continue

                # ================= CLEANING PIPELINE =================
                df = df.rename(columns={"date": "datetime"})
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

                # Sort is critical for RS windows.
                df = df.sort_values("datetime")

                # Remove duplicate dates.
                before_dupes = len(df)
                df = df.drop_duplicates(subset=["datetime"])
                dupes_removed = before_dupes - len(df)

                # Replace +/- infinity with NaN.
                df = df.replace([np.inf, -np.inf], np.nan)

                # Count NaN closes before removing.
                nan_count = df["close"].isna().sum()

                # Remove rows without close.
                df = df.dropna(subset=["close"])

                # Remove invalid close prices.
                invalid_price_count = (df["close"] <= 0).sum()
                df = df[df["close"] > 0]

                # Forward-fill small close gaps after invalid rows are removed.
                df["close"] = df["close"].ffill()

                final_rows = len(df)
                is_special = ticker in SPECIAL_TICKERS

                # Keep NSE stricter than USA. Only SPECIAL_TICKERS can bypass.
                if final_rows < MIN_VALID_ROWS and not is_special:
                    reason = f"Too few valid rows: {final_rows}"
                    skipped_tickers.append((ticker, reason))
                    batch_skipped_list.append(ticker)
                    batch_skipped += 1
                    continue

                if final_rows < 30:
                    if is_special:
                        logging.info(
                            f"✅ Force saved {ticker} "
                            f"with limited history ({final_rows} rows)"
                        )
                    else:
                        logging.warning(
                            f"⚠️ {ticker}: Limited history "
                            f"({final_rows} rows)"
                        )

                # Convert datetime to UNIX timestamp seconds.
                df["datetime"] = df["datetime"].astype("int64") // 10**9

                # ================= WRITE TO DB =================
                lib.write(ticker, df)
                success_tickers.append(ticker)
                batch_success += 1

                # ================= DEBUG LOGGING =================
                if nan_count > 0 or invalid_price_count > 0 or dupes_removed > 0:
                    logging.info(
                        f"{ticker}: cleaned → "
                        f"NaN_removed={nan_count}, "
                        f"invalid_price_removed={invalid_price_count}, "
                        f"duplicates_removed={dupes_removed}, "
                        f"final_rows={final_rows}"
                    )

            except Exception as e:
                failed_tickers.append((ticker, str(e)))
                batch_failed_list.append(ticker)
                batch_failed += 1
                logging.warning(f"❌ Failed {ticker}: {str(e)}")

        batch_time = time.time() - batch_start_time

        logging.info(
            f"✅ Batch {batch_no}/{total_batches} completed in {batch_time:.2f}s | "
            f"Success: {batch_success} | "
            f"Skipped: {batch_skipped} | "
            f"Failed: {batch_failed}"
        )

        if batch_skipped_list:
            logging.info(f"   Skipped tickers: {batch_skipped_list}")

        if batch_failed_list:
            logging.info(f"   Failed tickers: {batch_failed_list}")

    # ================= FINAL SUMMARY =================
    logging.info("\n=== FINAL FETCH SUMMARY ===")
    logging.info(f"Successful: {len(success_tickers)}")
    logging.info(f"Skipped: {len(skipped_tickers)}")
    logging.info(f"Failed: {len(failed_tickers)}")

    print(
        f"\n✅ Fetch complete! "
        f"Success: {len(success_tickers)}, "
        f"Skipped: {len(skipped_tickers)}, "
        f"Failed: {len(failed_tickers)}"
    )

    if skipped_tickers:
        print("\n--- Skipped Tickers ---")
        logging.info("--- Skipped Tickers ---")

        for ticker, reason in skipped_tickers:
            line = f"{ticker}: {reason}"
            print(line)
            logging.info(line)

    if failed_tickers:
        print("\n--- Failed Tickers ---")
        logging.info("--- Failed Tickers ---")

        for ticker, error in failed_tickers:
            line = f"{ticker}: {error}"
            print(line)
            logging.info(line)

    # Append final summary to the same log file for artifact visibility.
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(
            f"FINAL SUMMARY - Partition finished at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(
            f"Successful: {len(success_tickers)} | "
            f"Skipped: {len(skipped_tickers)} | "
            f"Failed: {len(failed_tickers)}\n"
        )

        if skipped_tickers:
            f.write("--- Skipped Tickers ---\n")
            for ticker, reason in skipped_tickers:
                f.write(f"{ticker}: {reason}\n")

        if failed_tickers:
            f.write("--- Failed Tickers ---\n")
            for ticker, error in failed_tickers:
                f.write(f"{ticker}: {error}\n")

        f.write("=" * 80 + "\n")

    print(f"Full log saved to: {log_file}")


def load_ticker_list(file_path, partition=None, total_partitions=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tickers = [item["ticker"] for item in data]

    if partition is not None and total_partitions:
        chunk_size = len(tickers) // total_partitions
        start = partition * chunk_size
        end = None if partition == total_partitions - 1 else start + chunk_size

        tickers = tickers[start:end]

        logging.info(
            f"Partition {partition}/{total_partitions}: "
            f"{len(tickers)} tickers"
        )

    return tickers


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Yahoo historical data and store in ArcticDB"
    )

    parser.add_argument(
        "input_file",
        help="Path to ticker_price.json"
    )

    parser.add_argument(
        "--log-file",
        default="logs/fetch_price_history_rs.log",
        help="Base log file name"
    )

    parser.add_argument(
        "--arctic-db-path",
        default="tmp/arctic_db",
        help="Directory for ArcticDB"
    )

    parser.add_argument(
        "--partition",
        type=int,
        default=None
    )

    parser.add_argument(
        "--total-partitions",
        type=int,
        default=None
    )

    args = parser.parse_args()

    # ================== DYNAMIC LOG FILE NAME ==================
    # This prevents parallel partitions from writing into the same log file.
    if args.partition is not None:
        base_name = os.path.splitext(
            os.path.basename(args.log_file)
        )[0]

        log_file = f"logs/{base_name}_{args.partition}.log"
    else:
        log_file = args.log_file

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.arctic_db_path, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filemode="a"
    )

    logging.info(
        f"Starting fetch for partition: "
        f"{args.partition if args.partition is not None else 'ALL'}"
    )

    tickers = load_ticker_list(
        args.input_file,
        args.partition,
        args.total_partitions
    )

    arctic = adb.Arctic(f"lmdb://{args.arctic_db_path}")

    fetch_historical_data(
        tickers,
        arctic,
        log_file
    )


if __name__ == "__main__":
    main()
