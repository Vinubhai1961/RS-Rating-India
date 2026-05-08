import pandas as pd
import os
import json
from datetime import datetime

# ==================== CONFIG ====================
input_path = 'source/India_Tickers_All.csv'          # Updated path
output_dir = 'data'
logs_dir = 'logs'
output_json_path = os.path.join(output_dir, 'ticker_info.json')
log_file = f"ticker_info_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_path = os.path.join(logs_dir, log_file)

# Ensure directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# ==================== LOAD DATA ====================
df = pd.read_csv(input_path)

# Log initial stats
initial_total_tickers = len(df)
nse_tickers = len(df[df['Exchange'] == 'NSE'])
bse_tickers = len(df[df['Exchange'] == 'BSE'])

print(f"Loaded {initial_total_tickers} tickers from source.")

# ==================== COLUMN NAME NORMALIZATION ====================
# Standardize column names (handle case and spacing differences)
df.columns = [col.strip() for col in df.columns]

column_mapping = {
    'Price change % 1 day': 'Price Change % 1 day',
    'Price Change % 1 day': 'Price Change % 1 day',
    'Relative volume 1 day': 'Relative Volume 1 day',
    'Relative Volume 1 day': 'Relative Volume 1 day',
    'Free float %': 'Free float %',
    'Performance % Year to date': 'Performance % Year to date',
}

for old_name, new_name in column_mapping.items():
    if old_name in df.columns:
        df = df.rename(columns={old_name: new_name})

# ==================== CLEANING FUNCTIONS ====================
def clean_percentage(col_name):
    if col_name not in df.columns:
        print(f"Warning: Column '{col_name}' not found. Skipping.")
        return None
    
    col = df[col_name]
    if col.dtype == 'object':
        # Remove % sign and convert to float
        col = col.astype(str).str.rstrip('%').replace('', pd.NA)
        col = pd.to_numeric(col, errors='coerce')
    df[col_name] = col
    return col

# Clean percentage columns
clean_percentage('Free float %')
clean_percentage('Performance % Year to date')
clean_percentage('Price Change % 1 day')
clean_percentage('Relative Volume 1 day')

# ==================== FILTERING ====================
# Filter out unwanted tickers
unwanted = ["Segregated Portfolio", "Segerated Portfolio", "Nippon India Equity Savings Fund"]
df = df[~df['Description'].str.contains('|'.join(unwanted), case=False, na=False)]

# ==================== DEDUPLICATION (NSE Priority) ====================
exchange_priority = {'NSE': 0, 'BSE': 1}
df['exchange_priority'] = df['Exchange'].map(exchange_priority).fillna(99)

df = df.sort_values(['Symbol', 'exchange_priority'])
df_unique = df.drop_duplicates(subset=['Symbol'], keep='first')

unique_tickers = len(df_unique)
df_unique = df_unique.drop(columns=['exchange_priority'])

print(f"After filtering and deduplication: {unique_tickers} unique tickers")

# ==================== CREATE TICKER COLUMN ====================
def get_ticker(row):
    if row['Exchange'] == 'NSE':
        return f"{row['Symbol']}.NS"
    elif row['Exchange'] == 'BSE':
        return f"{row['Symbol']}.BO"
    else:
        return f"{row['Symbol']}.{row['Exchange']}"

df_unique['Ticker'] = df_unique.apply(get_ticker, axis=1)

# ==================== SELECT & RENAME COLUMNS ====================
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

# Round numerical columns
columns_to_round = ['Price', '52WKH', '52WKL', 'FF', '1YR_Per', 'DPChange']
for col in columns_to_round:
    if col in output_df.columns:
        output_df[col] = output_df[col].round(2)

output_df = output_df.sort_values('Ticker')

# ==================== BUILD JSON ====================
json_data = []

# Add NIFTY 500 benchmark
json_data.append({
    "ticker": "^CRSLDX",
    "info": {
        "Ticker Name": "NIFTY 500",
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
    # Format DVol
    dvol = row['DVol']
    if pd.notnull(dvol):
        if dvol >= 1_000_000:
            dvol_formatted = f"{dvol / 1_000_000:.2f}M"
        elif dvol >= 1_000:
            dvol_formatted = f"{dvol / 1_000:.2f}K"
        else:
            dvol_formatted = f"{dvol:.0f}"
    else:
        dvol_formatted = None

    rvol = f"{row['RVol']:.2f}" if pd.notnull(row['RVol']) else None
    ff = f"{row['FF']:.2f}%" if pd.notnull(row['FF']) else None
    
    ytd = row['1YR_Per']
    ytd_formatted = f"+{ytd:.2f}%" if pd.notnull(ytd) and ytd >= 0 else f"{ytd:.2f}%" if pd.notnull(ytd) else None
    
    dpchange = f"{row['DPChange']:.2f}%" if pd.notnull(row['DPChange']) else None

    ticker_entry = {
        "ticker": row['Ticker'],
        "info": {
            "Ticker Name": row['Ticker Name'],
            "Price": row['Price'],
            "DVol": dvol_formatted,
            "RVol": rvol,
            "Sector": row['Sector'],
            "Industry": row['Industry'],
            "52WKH": row['52WKH'],
            "52WKL": row['52WKL'],
            "Exchange": row['Exchange'],
            "FF": ff,
            "1YR_Per": ytd_formatted,
            "DPChange": dpchange
        }
    }
    json_data.append(ticker_entry)

# ==================== SAVE OUTPUT ====================
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

# Write log
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(f"Log generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Initial total tickers: {initial_total_tickers}\n")
    f.write(f"NSE tickers: {nse_tickers}\n")
    f.write(f"BSE tickers: {bse_tickers}\n")
    f.write(f"Final unique tickers: {unique_tickers}\n")
    f.write(f"Output saved to: {output_json_path}\n")

print(f"✅ Success! Output JSON created at: {output_json_path}")
print(f"📊 Log file created at: {log_path}")
print(f"Total records in JSON: {len(json_data)}")
