import json
import datetime
from pathlib import Path

# Define input and output paths
input_path = "data/ticker_price.json"
output_dir = "25%_52WKH"
current_date = datetime.datetime.now().strftime("%Y%m%d")
output_path = f"{output_dir}/name_{current_date}.txt"

# Ensure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Initialize list for tickers within 25% of 52-week high
tickers_within_25percent = []

try:
    # Read the JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Process each ticker
    for item in data:
        try:
            ticker = item['ticker']
            price = float(item['info']['Price'])
            wk_high = float(item['info']['52WKH'])
            
            # Calculate 75% of 52-week high (threshold for being within 25%)
            threshold = wk_high * 0.75
            
            # Check if current price is within 25% of 52-week high
            if price >= threshold:
                # Remove .NS or .BO from ticker
                clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
                tickers_within_25percent.append(clean_ticker)
        except (KeyError, ValueError) as e:
            print(f"Skipping ticker {item.get('ticker', 'unknown')}: Invalid data - {e}")
            continue

    # Write the tickers to the output file
    with open(output_path, 'w') as file:
        if tickers_within_25percent:
            file.write(', '.join(tickers_within_25percent))
        else:
            file.write("No tickers found within 25% of 52-week high.")

    print(f"Output written to {output_path}")

except FileNotFoundError:
    print(f"Error: Input file {input_path} not found.")
except json.JSONDecodeError as e:
    print(f"Error: Failed to parse JSON file - {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
