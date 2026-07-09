#!/usr/bin/env python3

import os
from datetime import datetime

import pandas_market_calendars as mcal
import pytz


def is_india_trading_day():
    """
    Returns True if NSE/BSE are open today.
    """

    try:
        nse = mcal.get_calendar("NSE")

        today = datetime.now(
            pytz.timezone("Asia/Kolkata")
        ).date()

        schedule = nse.schedule(
            start_date=today.isoformat(),
            end_date=today.isoformat()
        )

        is_open = len(schedule) > 0

        print(
            f"NSE trading day check for {today}: "
            f"{'OPEN' if is_open else 'CLOSED'}"
        )

        return is_open

    except Exception as e:

        print(f"Calendar error: {e}")

        weekday = datetime.now(
            pytz.timezone("Asia/Kolkata")
        ).weekday()

        is_open = weekday < 5

        print(
            f"Fallback (Mon-Fri only): "
            f"{'OPEN' if is_open else 'CLOSED'}"
        )

        return is_open


if __name__ == "__main__":

    market_open = is_india_trading_day()

    output = os.getenv("GITHUB_OUTPUT")

    if output:
        with open(output, "a") as f:
            f.write(f"market_open={str(market_open).lower()}\n")
            f.write(f"should_run={str(market_open).lower()}\n")
    else:
        print(f"Manual run: {market_open}")
