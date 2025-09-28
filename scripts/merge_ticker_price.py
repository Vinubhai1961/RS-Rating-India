#!/usr/bin/env python3
import os
import json
import argparse
import logging
import time
import re
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/merge_ticker_price.log"), logging.StreamHandler()]
)

def merge_price_files(artifacts_dir, expected_parts=None):
    output_file = os.path.join("data", "ticker_price.json")
    merged_data = []

    if not os.path.exists(artifacts_dir):
        logging.error(f"Input directory {artifacts_dir} does not exist")
        return

    logging.info(f"Searching for ticker_price_part_*.json files in {artifacts_dir}")
    part_files = sorted([f for f in os.listdir(artifacts_dir) if f.startswith("ticker_price_part_") and f.endswith(".json")])
    if not part_files:
        logging.error(f"No ticker_price_part_*.json files found in {artifacts_dir}")
        return

    logging.info(f"Found {len(part_files)} part files to merge: {part_files}")
    if expected_parts is not None and len(part_files) < expected_parts:
        logging.warning(f"Expected {expected_parts} part files, but found only {len(part_files)}")

    # Regex to validate volume formats (e.g., "102.45K", "7.80M", "7.80B")
    volume_pattern = r"^\d+\.?\d{0,2}[KMB]?$"

    for filename in part_files:
        file_path = os.path.join(artifacts_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                part_data = json.load(f)
                logging.info(f"Loaded {len(part_data)} tickers from {filename}")
                for item in part_data:
                    if not isinstance(item, dict) or "ticker" not in item or "info" not in item:
                        logging.warning(f"Invalid data in {filename}: missing 'ticker' or 'info'")
                        continue
                    info = item["info"]
                    required_fields = [
                        "Ticker Name", "Price", "DVol", "RVol", "Sector", "Industry", "type",
                        "52WKL", "52WKH", "MCAP", "AvgVol", "AvgVol10", "Exchange", "FF", "1YR_Per", "DPChange"
                    ]
                    missing_fields = [f for f in required_fields if f not in info]
                    if missing_fields:
                        logging.warning(f"Missing fields for {item['ticker']} in {filename}: {missing_fields}")
                        continue
                    
                    # Validate numerical fields
                    if not isinstance(info["Price"], (int, float)) or info["Price"] <= 0:
                        logging.warning(f"Invalid Price for {item['ticker']} in {filename}: {info['Price']}")
                        continue
                    if info["52WKL"] is not None and (not isinstance(info["52WKL"], (int, float)) or info["52WKL"] <= 0):
                        logging.warning(f"Invalid 52WKL for {item['ticker']} in {filename}: {info['52WKL']}")
                        continue
                    if info["52WKH"] is not None and (not isinstance(info["52WKH"], (int, float)) or info["52WKH"] <= 0):
                        logging.warning(f"Invalid 52WKH for {item['ticker']} in {filename}: {info['52WKH']}")
                        continue
                    
                    # Validate formatted volume and market cap fields
                    for field in ["DVol", "AvgVol", "AvgVol10"]:
                        if info[field] is not None and (not isinstance(info[field], str) or not re.match(volume_pattern, info[field])):
                            logging.warning(f"Invalid {field} for {item['ticker']} in {filename}: {info[field]}")
                            continue
                    if info["MCAP"] is not None and (not isinstance(info["MCAP"], str) or not re.match(volume_pattern, info["MCAP"])):
                        logging.warning(f"Invalid MCAP for {item['ticker']} in {filename}: {info['MCAP']}")
                        continue
                    
                    # Validate string fields
                    if not isinstance(info["Ticker Name"], str):
                        logging.warning(f"Invalid Ticker Name for {item['ticker']} in {filename}: {info['Ticker Name']}")
                        continue
                    if not isinstance(info["Sector"], str):
                        logging.warning(f"Invalid Sector for {item['ticker']} in {filename}: {info['Sector']}")
                        continue
                    if not isinstance(info["Industry"], str):
                        logging.warning(f"Invalid Industry for {item['ticker']} in {filename}: {info['Industry']}")
                        continue
                    if not isinstance(info["type"], str):
                        logging.warning(f"Invalid type for {item['ticker']} in {filename}: {info['type']}")
                        continue
                    if not isinstance(info["Exchange"], str):
                        logging.warning(f"Invalid Exchange for {item['ticker']} in {filename}: {info['Exchange']}")
                        continue
                    if info["RVol"] is not None and not isinstance(info["RVol"], str):
                        logging.warning(f"Invalid RVol for {item['ticker']} in {filename}: {info['RVol']}")
                        continue
                    if info["FF"] is not None and not isinstance(info["FF"], str):
                        logging.warning(f"Invalid FF for {item['ticker']} in {filename}: {info['FF']}")
                        continue
                    if info["1YR_Per"] is not None and not isinstance(info["1YR_Per"], str):
                        logging.warning(f"Invalid 1YR_Per for {item['ticker']} in {filename}: {info['1YR_Per']}")
                        continue
                    if info["DPChange"] is not None and not isinstance(info["DPChange"], str):
                        logging.warning(f"Invalid DPChange for {item['ticker']} in {filename}: {info['DPChange']}")
                        continue
                    
                    merged_data.append(item)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse {filename}: {e}")
        except Exception as e:
            logging.error(f"Error reading {filename}: {e}")

    if not merged_data:
        logging.error("No valid data merged from part files. Skipping output file creation.")
        return

    os.makedirs("data", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)
    logging.info(f"Merged data saved to {output_file} with {len(merged_data)} entries")

def main(artifacts_dir, expected_parts=None):
    start_time = time.time()
    start_time_str = datetime.now().strftime("%I:%M %p EDT on %A, %B %d, %Y")
    logging.info(f"Starting price merge process at {start_time_str}")

    merge_price_files(artifacts_dir, expected_parts)

    elapsed_time = time.time() - start_time
    logging.info(f"Price merge completed. Elapsed time: {elapsed_time:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ticker price partition files into a single JSON file.")
    parser.add_argument("artifacts_dir", help="Directory containing ticker price partition files")
    parser.add_argument("--part-total", type=int, default=None, help="Expected number of part files (optional)")
    args = parser.parse_args()

    main(args.artifacts_dir, args.part_total)
