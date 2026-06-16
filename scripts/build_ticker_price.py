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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)

def normalize_yahoo_symbol(symbol):
    if not symbol:
        return symbol
    return symbol.replace("_", "-")

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
        return int(float(dvol))
    except Exception:
        return None

def load_ticker_info():
    if not os.path.exists(TICKER_INFO_FILE):
        return {}
    with open(TICKER_INFO_FILE, "r", encoding="utf-8") as f:
        return {
            item["ticker"]: {
                "info": {
                    k: parse_dvol(v) if k == "DVol" else v
                    for k, v in item["info"].items()
                }
            }
            for item in json.load(f)
        }

def partition_tickers(tickers, part_index, part_total):
    per_part = len(tickers) // part_total
    start = part_index * per_part
    end = start + per_part if part_index < part_total - 1 else len(tickers)
    return tickers[start:end]

def format_volume(value):
    if not value or value <= 0:
        return None
    return f"{value / 1000:.2f}K"

def format_market_cap(value):
    if not value or value <= 0:
        return None
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    return f"{value / 1_000_000:.2f}M"

def process_batch(batch, ticker_info, is_volume_retry=False):

    for attempt in range(MAX_BATCH_RETRIES):

        try:
            prices = []
            failure_reasons = {"no_price":0,"below_threshold":0,"missing_volume":0,"error":0}

            symbol_map = {normalize_yahoo_symbol(s): s for s in batch}
            yahoo_batch = list(symbol_map.keys())

            yq = Ticker(yahoo_batch, validate=False)

            try:
                hist = yq.history(period="1d")
            except Exception as e:
                raise RuntimeError(f"history failed: {e}")

            try:
                summary_details = yq.summary_detail
            except Exception:
                summary_details = {}

            for yahoo_symbol in yahoo_batch:

                symbol = symbol_map[yahoo_symbol]

                try:
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
                    except Exception:
                        pass

                    summary = {}
                    if isinstance(summary_details, dict):
                        sym_summary = (
                            summary_details.get(yahoo_symbol)
                            or summary_details.get(symbol)
                            or {}
                        )
                        if isinstance(sym_summary, dict):
                            summary = sym_summary

                    if isinstance(price, str):
                        try:
                            price = float(price)
                        except Exception:
                            price = None

                    if price is None:
                        for key in [
                            "regularMarketPrice",
                            "previousClose",
                            "currentPrice",
                            "price",
                            "open"
                        ]:
                            value = summary.get(key)
                            if isinstance(value, (int, float)):
                                price = value
                                break

                    if price is None or not isinstance(price, (int, float)):
                        failure_reasons["no_price"] += 1
                        continue

                    if price < PRICE_THRESHOLD:
                        failure_reasons["below_threshold"] += 1
                        continue

                    volume = summary.get("volume")
                    avg_volume = summary.get("averageVolume")
                    avg_volume_10days = (
                        summary.get("averageVolume10days")
                        or summary.get("averageDailyVolume10Day")
                        or summary.get("averageVolume")
                    )

                    if not avg_volume_10days:
                        failure_reasons["missing_volume"] += 1

                    info = ticker_info.get(symbol, {}).get("info", {})

                    rvol = None
                    if volume and avg_volume_10days:
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
                            "52WKL": summary.get("fiftyTwoWeekLow"),
                            "52WKH": summary.get("fiftyTwoWeekHigh"),
                            "MCAP": format_market_cap(summary.get("marketCap")),
                            "AvgVol": format_volume(avg_volume),
                            "AvgVol10": format_volume(avg_volume_10days),
                            "Exchange": info.get("Exchange", "n/a"),
                            "FF": info.get("FF"),
                            "1YR_Per": info.get("1YR_Per"),
                            "DPChange": info.get("DPChange")
                        }
                    })

                except Exception:
                    failure_reasons["error"] += 1

            failed_tickers = [s for s in batch if s not in [p["ticker"] for p in prices]]

            logging.info(f"Batch {'[Volume Retry]' if is_volume_retry else ''} stats: {failure_reasons}")
            return len(prices), failed_tickers, prices

        except Exception as e:
            logging.warning(f"Batch failed ({attempt+1}/{MAX_BATCH_RETRIES}): {e}")
            if attempt < MAX_BATCH_RETRIES - 1:
                time.sleep(random.uniform(5,10))

    return 0, batch, []

# Main body intentionally same as user version except retry-pass enhancement.
