import json
import datetime
from pathlib import Path

# Define input and output paths
input_path = "data/ticker_price.json"
output_dir = "25%_52WKH"
current_date = datetime.datetime.now().strftime("%Y%m%d")
output_path = f"{output_dir}/25%_52wkh_{current_date}.txt"

# Ensure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Initialize list for tickers within 25% of 52-week high and RVol > 1
tickers_within_25percent = []

try:
    # Read the JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Process each ticker
    for item in data:
        try:
            ticker = item['ticker']
            price = item['info']['Price']
            wk_high = item['info']['52WKH']
            rvol = item['info']['RVol']
            
            # Check for None or invalid values
            if price is None or wk_high is None or rvol is None:
                print(f"Skipping ticker {ticker}: Missing Price, 52WKH, or RVol")
                continue
            
            # Convert to float and handle potential conversion errors
            try:
                price = float(price)
                wk_high = float(wk_high)
                rvol = float(rvol)
            except (ValueError, TypeError) as e:
                print(f"Skipping ticker {ticker}: Invalid data - {e}")
                continue
            
            # Calculate 75% of 52-week high (threshold for being within 25%)
            threshold = wk_high * 0.75
            
            # Check if current price is within 25% of 52-week high and RVol > 1
            if price >= threshold and rvol > 0.5:
                # Remove .NS or .BO from ticker
                clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
                tickers_within_25percent.append(clean_ticker)
        except KeyError as e:
            print(f"Skipping ticker {item.get('ticker', 'unknown')}: Missing key - {e}")
            continue

    # Write the tickers to the output file, with up to 995 tickers per line
    with open(output_path, 'w') as file:
        if tickers_within_25percent:
            # Split tickers into chunks of 995
            chunk_size = 995
            for i in range(0, len(tickers_within_25percent), chunk_size):
                chunk = tickers_within_25percent[i:i + chunk_size]
                file.write(', '.join(chunk) + '\n')
        else:
            file.write("No tickers found within 25% of 52-week high with RVol > 1.")

    print(f"Output written to {output_path}")

except FileNotFoundError:
    print(f"Error: Input file {input_path} not found.")
except json.JSONDecodeError as e:
    print(f"Error: Failed to parse JSON file - {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
