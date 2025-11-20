import pandas as pd
import os
import json
from datetime import datetime

# Define paths based on repo structure
input_path = 'source/India_Tickers_All.csv'
output_dir = 'data'
logs_dir = 'logs'
output_json_path = os.path.join(output_dir, 'ticker_info.json')
log_file = f"ticker_info_{datetime.now().strftime('%Y-%m-%d')}.log"
log_path = os.path.join(logs_dir, log_file)

# Ensure output and logs directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Read the input CSV
df = pd.read_csv(input_path)

# Log initial total tickers
initial_total_tickers = len(df)

# Count tickers by exchange before filtering
nse_tickers = len(df[df['Exchange'] == 'NSE'])
bse_tickers = len(df[df['Exchange'] == 'BSE'])

# Filter out unwanted tickers:
# Description containing "Segregated Portfolio", "Segerated Portfolio", or "Nippon India Equity Savings Fund"
df = df[
    ~df['Description'].str.contains(
        "Segregated Portfolio|Segerated Portfolio|Nippon India Equity Savings Fund",
        case=False, na=False
    )
]

# Clean numerical columns: strip '%' if present and convert to float
def clean_percentage(col):
    if df[col].dtype == 'object':
        df[col] = df[col].str.rstrip('%').astype(float)
    return df[col]

df['Free float %'] = clean_percentage('Free float %')
df['Performance % Year to date'] = clean_percentage('Performance % Year to date')
df['Price Change % 1 day'] = clean_percentage('Price Change % 1 day')

# Prioritize NSE: sort by Symbol, then by Exchange priority (NSE first)
exchange_priority = {'NSE': 0, 'BSE': 1}  # Default to 99 for others
df['exchange_priority'] = df['Exchange'].map(exchange_priority).fillna(99)

# Sort and drop duplicates, keeping the first (highest priority)
df = df.sort_values(['Symbol', 'exchange_priority'])
df_unique = df.drop_duplicates(subset=['Symbol'], keep='first')

# Count unique tickers in final output
unique_tickers = len(df_unique)

# Drop the temporary column
df_unique = df_unique.drop(columns=['exchange_priority'])

# Create 'Ticker' column
def get_ticker(row):
    if row['Exchange'] == 'NSE':
        return f"{row['Symbol']}.NS"
    elif row['Exchange'] == 'BSE':
        return f"{row['Symbol']}.BO"
    else:
        return f"{row['Symbol']}.{row['Exchange']}"  # Fallback for other exchanges

df_unique['Ticker'] = df_unique.apply(get_ticker, axis=1)

# Select and rename columns
output_df = df_unique[[
    'Ticker', 'Description', 'Price', 'Volume 1 day', 'Relative Volume 1 day',
    'Sector', 'Industry', 'High 52 weeks', 'Low 52 weeks', 'Exchange',
    'Free float %', 'Performance % Year to date', 'Price Change % 1 day'
]].rename(columns={
    'Description': 'Ticker Name',
    'Volume 1 day': 'DVol',
    'Relative Volume 1 day': 'RVol',
    'High 52 weeks': '52WKH',
    'Low 52 weeks': '52WKL',
    'Free float %': 'FF',
    'Performance % Year to date': '1YR_Per',
    'Price Change % 1 day': 'DPChange'
})

# Round specified columns to 2 decimals
columns_to_round = ['Price', '52WKH', '52WKL', 'FF', '1YR_Per', 'DPChange']
for col in columns_to_round:
    output_df[col] = output_df[col].round(2)

# Sort by Ticker for consistent output
output_df = output_df.sort_values('Ticker')

# Prepare JSON output
json_data = []

# Add NIFTY 500 benchmark ticker as the first record
json_data.append({
    "ticker": "^NSEI",
    "info": {
        "Ticker Name": "NIFTY 50",
        "Price": "",
        "DVol": "",
        "RVol": "",
        "Sector": "",
        "Industry": "",
        "52WKH": None,
        "52WKL": None,
        "Exchange": "",
        "FF": None,
        "1YR_Per": None,
        "DPChange": None
    }
})

for _, row in output_df.iterrows():
    # Format DVol based on magnitude
    dvol_value = row['DVol']
    if pd.notnull(dvol_value):
        if dvol_value >= 1_000_000:
            dvol_formatted = f"{dvol_value / 1_000_000:.2f}M"  # e.g., 10,997,211 → 11.00M
        elif dvol_value >= 1_000:
            dvol_formatted = f"{dvol_value / 1_000:.2f}K"  # e.g., 498,949 → 498.95K
        else:
            dvol_formatted = f"{dvol_value:.0f}"  # e.g., 500 → 500
    else:
        dvol_formatted = None

    # Format RVol to 2 decimals (e.g., 8.591089492488486 to 8.59)
    rvol_value = row['RVol']
    rvol_formatted = f"{rvol_value:.2f}" if pd.notnull(rvol_value) else None

    # Format FF as percentage (e.g., 30.03 to 30.03%)
    ff_value = row['FF']
    ff_formatted = f"{ff_value:.2f}%" if pd.notnull(ff_value) else None

    # Format 1YR_Per as signed percentage (e.g., -82.81 to -82.81%)
    ytd_value = row['1YR_Per']
    ytd_formatted = f"+{ytd_value:.2f}%" if pd.notnull(ytd_value) and ytd_value >= 0 else f"{ytd_value:.2f}%"

    # Format DPChange as percentage (e.g., 893.9504325765956 to 893.95%)
    dpchange_value = row['DPChange']
    dpchange_formatted = f"{dpchange_value:.2f}%" if pd.notnull(dpchange_value) else None

    ticker_entry = {
        "ticker": row['Ticker'],
        "info": {
            "Ticker Name": row['Ticker Name'],
            "Price": row['Price'],
            "DVol": dvol_formatted,
            "RVol": rvol_formatted,
            "Sector": row['Sector'],
            "Industry": row['Industry'],
            "52WKH": row['52WKH'],
            "52WKL": row['52WKL'],
            "Exchange": row['Exchange'],
            "FF": ff_formatted,
            "1YR_Per": ytd_formatted,
            "DPChange": dpchange_formatted
        }
    }
    json_data.append(ticker_entry)

# Write to output JSON
with open(output_json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

# Write to log file
with open(log_path, 'w') as f:
    f.write(f"Log generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Initial total tickers from source: {initial_total_tickers}\n")
    f.write(f"Total tickers from NSE: {nse_tickers}\n")
    f.write(f"Total tickers from BSE: {bse_tickers}\n")
    f.write(f"Unique tickers in final output: {unique_tickers}\n")

print(f"Output JSON created at {output_json_path}")
print(f"Log file created at {log_path}")
