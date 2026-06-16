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
MAX_VOLUME_RETRIES = 2              # Dedicated volume retry attempts
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

    # Avoid duplicate handlers when basicConfig already ran.
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(LOG_PATH, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )


def normalize_yahoo_symbol(symbol):
    """
    Convert internal NSE/BSE symbols to Yahoo-compatible symbols.

    Example:
        BAJAJ_AUTO.NS -> BAJAJ-AUTO.NS

    Output JSON still keeps the original symbol from ticker_info.json.
    """
    if not symbol:
        return symbol

    return str(symbol).replace("_", "-")


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
            return {
                item["ticker"]: {
                    "info": {
                        k: parse_dvol(v) if k == "DVol" else v
                        for k, v in item["info"].items()
                    }
                }
                for item in json.load(f)
            }
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


def get_summary_for_symbol(summary_details, yahoo_symbol, original_symbol):
    """
    YahooQuery usually keys summary_detail by Yahoo symbol.
    Keep original-symbol fallback for safety.
    """
    if not isinstance(summary_details, dict):
        return {}

    sym_summary = (
        summary_details.get(yahoo_symbol)
        or summary_details.get(original_symbol)
        or {}
    )

    # Yahoo sometimes returns error strings instead of dict objects.
    if isinstance(sym_summary, str):
        logging.debug(f"Invalid summary for {original_symbol}: {sym_summary}")
        return {}

    if isinstance(sym_summary, dict):
        return sym_summary

    return {}


def extract_price_from_history(hist, yahoo_symbol, original_symbol):
    """Extract last close from Yahoo history dataframe safely."""
    price = None

    try:
        if hasattr(hist, "index"):
            symbols_in_hist = hist.index.get_level_values(0)

            if yahoo_symbol in symbols_in_hist:
                symbol_hist = hist.loc[yahoo_symbol]

                if not symbol_hist.empty:
                    close_series = symbol_hist.get("close")

                    if close_series is not None and len(close_series) > 0:
                        price = close_series.iloc[-1]

    except Exception as hist_err:
        logging.debug(f"{original_symbol} history parse failed: {hist_err}")

    if isinstance(price, str):
        try:
            price = float(price)
        except Exception:
            price = None

    return price


def extract_price_from_summary(summary, original_symbol):
    """
    USA-style price fallback chain.
    Used only when history did not return a usable close.
    """
    for key in [
        "regularMarketPrice",
        "previousClose",
        "currentPrice",
        "price",
        "open"
    ]:
        value = summary.get(key)

        if isinstance(value, (int, float)):
            logging.debug(f"{original_symbol} recovered price from summary.{key} = {value}")
            return value

    return None


def process_batch(batch, ticker_info, is_volume_retry=False):
    """
    Process one batch with:
      - Yahoo symbol normalization: BAJAJ_AUTO.NS -> BAJAJ-AUTO.NS
      - Batch-level retry wrapper
      - History price extraction
      - Summary-detail fallback price chain
      - AvgVol10 fallback chain
    """

    for attempt in range(MAX_BATCH_RETRIES):
        prices = []
        failure_reasons = {
            "no_price": 0,
            "below_threshold": 0,
            "missing_volume": 0,
            "error": 0
        }

        try:
            # Map Yahoo-compatible symbol back to original ticker_info symbol.
            symbol_map = {}
            for symbol in batch:
                yahoo_symbol = normalize_yahoo_symbol(symbol)

                # If two symbols normalize to the same Yahoo symbol, keep the first
                # and log the duplicate. This should be rare.
                if yahoo_symbol in symbol_map and symbol_map[yahoo_symbol] != symbol:
                    logging.debug(
                        f"Duplicate normalized symbol {yahoo_symbol}: "
                        f"{symbol_map[yahoo_symbol]} and {symbol}"
                    )
                else:
                    symbol_map[yahoo_symbol] = symbol

            yahoo_batch = list(symbol_map.keys())

            normalized_count = sum(
                1 for yahoo_symbol, original_symbol in symbol_map.items()
                if yahoo_symbol != original_symbol
            )

            if normalized_count:
                logging.debug(
                    f"Normalized {normalized_count} symbols in batch. "
                    f"Sample: {[(v, k) for k, v in list(symbol_map.items())[:10] if k != v]}"
                )

            # Keep original NSE logic, but query Yahoo-compatible tickers.
            yq = Ticker(yahoo_batch, validate=False)

            try:
                hist = yq.history(period="1d")
            except Exception as e:
                raise RuntimeError(f"History fetch failed for batch: {e}")

            try:
                summary_details = yq.summary_detail
            except Exception as e:
                logging.warning(f"summary_detail fetch failed: {e}")
                summary_details = {}

            for yahoo_symbol in yahoo_batch:
                symbol = symbol_map[yahoo_symbol]

                try:
                    # -----------------------------
                    # PRICE EXTRACTION
                    # -----------------------------
                    price = extract_price_from_history(hist, yahoo_symbol, symbol)

                    # -----------------------------
                    # SUMMARY DETAIL
                    # -----------------------------
                    summary = get_summary_for_symbol(summary_details, yahoo_symbol, symbol)

                    # USA-style price fallback chain.
                    if price is None or not isinstance(price, (int, float)):
                        price = extract_price_from_summary(summary, symbol)

                    if price is None or not isinstance(price, (int, float)):
                        failure_reasons["no_price"] += 1
                        continue

                    if price < PRICE_THRESHOLD:
                        failure_reasons["below_threshold"] += 1
                        continue

                    volume = summary.get("volume")
                    avg_volume = summary.get("averageVolume")

                    # AvgVol10 fallback chain.
                    # Some Yahoo payloads use averageDailyVolume10Day, and some NSE/SME
                    # tickers only expose averageVolume.
                    avg_volume_10days = (
                        summary.get("averageVolume10days")
                        or summary.get("averageDailyVolume10Day")
                        or summary.get("averageVolume")
                    )

                    if avg_volume_10days is None or avg_volume_10days <= 0:
                        failure_reasons["missing_volume"] += 1
                        logging.debug(
                            f"{symbol}: AvgVol10 missing. "
                            f"Yahoo symbol={yahoo_symbol}, "
                            f"summary keys={list(summary.keys())[:25]}"
                        )

                    fifty_two_week_low = summary.get("fiftyTwoWeekLow")
                    fifty_two_week_high = summary.get("fiftyTwoWeekHigh")
                    market_cap = summary.get("marketCap")

                    info = ticker_info.get(symbol, {}).get("info", {})

                    rvol = None
                    if volume is not None and avg_volume_10days is not None and avg_volume_10days > 0:
                        rvol = f"{volume / avg_volume_10days:.2f}"
                        
                    sector = info.get("Sector", "n/a")
                    industry = info.get("Industry", "n/a")
                    
                    sector = str(sector).title() if sector is not None else "N/A"
                    industry = str(industry).title() if industry is not None else "N/A"

                    prices.append({
                        "ticker": symbol,
                        "info": {
                            "Ticker Name": info.get("Ticker Name", "n/a"),
                            "Price": round(price, 2),
                            "DVol": format_volume(volume),
                            "RVol": rvol,
                            "Sector": Sector,
                            "Industry": industry,
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

            processed = {p["ticker"] for p in prices}
            failed_tickers = [s for s in batch if s not in processed]

            logging.info(
                f"Batch {'[Volume Retry]' if is_volume_retry else ''} "
                f"stats: {failure_reasons}"
            )

            return len(prices), failed_tickers, prices

        except Exception as e:
            if "429" in str(e) or "curl" in str(e).lower():
                logging.warning(
                    f"Rate limit / curl issue in batch "
                    f"(attempt {attempt + 1}/{MAX_BATCH_RETRIES}): {e}"
                )
            else:
                logging.warning(
                    f"Batch failed "
                    f"(attempt {attempt + 1}/{MAX_BATCH_RETRIES}): {e}"
                )

            if attempt < MAX_BATCH_RETRIES - 1:
                sleep_time = random.uniform(5, 10)
                logging.info(f"Retrying batch after {sleep_time:.1f}s")
                time.sleep(sleep_time)
            else:
                logging.error(
                    f"Batch permanently failed after {MAX_BATCH_RETRIES} attempts"
                )

    return 0, batch, []


def merge_prices_by_ticker(all_prices, new_prices):
    """
    Merge new prices into all_prices by ticker.
    Existing tickers are replaced; new tickers are appended.
    """
    index = {p["ticker"]: i for i, p in enumerate(all_prices)}

    for np in new_prices:
        ticker = np["ticker"]

        if ticker in index:
            all_prices[index[ticker]] = np
        else:
            index[ticker] = len(all_prices)
            all_prices.append(np)

    return all_prices


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

        logging.info(f"Batch {idx}/{len(batches)} - Fetched data for {updated} tickers")

        # Collect tickers missing AvgVol10 for retry
        for p in prices:
            if p["info"].get("AvgVol10") is None:
                volume_missing_tickers.append(p["ticker"])

        if idx < len(batches):
            delay = random.uniform(*BATCH_DELAY_RANGE)
            time.sleep(delay)

    # Dedicated Volume Retry Pass
    if volume_missing_tickers:
        volume_missing_tickers = sorted(set(volume_missing_tickers))
        logging.info(f"Starting dedicated volume retry for {len(volume_missing_tickers)} tickers...")
        logging.info(f"Volume-missing tickers sample: {volume_missing_tickers[:50]}")

        with open(f"logs/volume_missing_part_{part_index}.txt", "w", encoding="utf-8") as vf:
            for ticker in volume_missing_tickers:
                vf.write(f"{ticker}\n")

        vol_batches = [
            volume_missing_tickers[i:i + BATCH_SIZE]
            for i in range(0, len(volume_missing_tickers), BATCH_SIZE)
        ]

        for vidx, vbatch in enumerate(tqdm(vol_batches, desc="Volume Retry"), 1):
            _, _, vol_prices = process_batch(vbatch, ticker_info, is_volume_retry=True)

            # Merge better data while preserving the same output schema.
            all_prices = merge_prices_by_ticker(all_prices, vol_prices)

            if vidx < len(vol_batches):
                time.sleep(random.uniform(*VOLUME_RETRY_DELAY))

    # Retry unresolved tickers.
    # Your previous logging patch only wrote unresolved tickers; this now actually reprocesses them.
    unresolved_unique = sorted(set(all_failed))

    if unresolved_unique:
        logging.info(f"Unresolved ticker sample: {unresolved_unique[:50]}")

        with open(UNRESOLVED_PRICE_TICKERS % part_index, "w", encoding="utf-8") as f:
            for ticker in unresolved_unique:
                f.write(f"{ticker}\n")

        logging.info(f"Retry sub-pass for {len(unresolved_unique)} unresolved tickers")

        retry_batches = [
            unresolved_unique[i:i + BATCH_SIZE]
            for i in range(0, len(unresolved_unique), BATCH_SIZE)
        ]

        retry_recovered = 0
        retry_still_failed = []

        for ridx, retry_batch in enumerate(tqdm(retry_batches, desc="Retry Pass"), 1):
            updated, failed, retry_prices = process_batch(retry_batch, ticker_info)
            retry_recovered += updated
            retry_still_failed.extend(failed)

            all_prices = merge_prices_by_ticker(all_prices, retry_prices)

            if ridx < len(retry_batches):
                time.sleep(random.uniform(5, 10))

        retry_still_failed = sorted(set(retry_still_failed))

        with open(UNRESOLVED_PRICE_TICKERS % part_index, "w", encoding="utf-8") as f:
            for ticker in retry_still_failed:
                f.write(f"{ticker}\n")

        logging.info(
            f"Retry pass complete. Recovered {retry_recovered} tickers. "
            f"Still unresolved: {len(retry_still_failed)}"
        )

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
