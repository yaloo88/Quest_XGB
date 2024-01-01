import os
from datetime import datetime
import pytz
import csv
from collections import deque
import json
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

def extract_symbol_interval(filename):
    parts = filename.split('_')
    symbol = parts[0]
    interval = parts[1].split('.')[0]  # Remove '.csv'
    return symbol, interval

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_symbol_info(filename):
    data_folder = os.path.join('data')
    candles_folder = os.path.join(data_folder, 'candles')
    symbols_folder = os.path.join(data_folder, 'symbols')

    csv_filename = filename
    candles_file_path = os.path.join(candles_folder, csv_filename)

    if not os.path.exists(candles_file_path):
        print(f"File {csv_filename} does not exist in the directory.")
        return None
    else:
        symbol, interval = extract_symbol_interval(csv_filename)

        with open(candles_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            last_row = deque(reader, maxlen=1)[0]
        end_value = last_row[1]  # Last 'end' value

        symbol_json_file = f"{symbol}.json"
        symbol_file_path = os.path.join(symbols_folder, symbol_json_file)

        if os.path.exists(symbol_file_path):
            symbol_data = read_json_file(symbol_file_path)
            # Extract JSON data
            symbol_description = symbol_data['description']
            symbol_id = symbol_data['symbolId']
            security_type = symbol_data['securityType']
            listing_exchange = symbol_data['listingExchange']
            is_tradable = symbol_data['isTradable']
            is_quotable = symbol_data['isQuotable']
            currency = symbol_data['currency']

            # Return the extracted data
            return symbol, interval, end_value, symbol_description, symbol_id, security_type, listing_exchange, is_tradable, is_quotable, currency

        else:
            print(f"No JSON data available for {symbol}")
            return symbol, interval, end_value

def get_market_candles(symbol_id, interval,end_value):
    from api_auth import get_api_key_data, make_api_call

    # Retrieve API key data
    api_key_data = get_api_key_data()
    if api_key_data:
        # Accessing the access_token and server URL from the dictionary
        access_token = api_key_data.get('access_token')
        server_url = api_key_data.get('api_server')

        if access_token and server_url:
            # Calculate start and end times
            # Get the current time in the specified timezone
            timezone = pytz.timezone('America/New_York')
            current_time = datetime.now(timezone)
            start_time = end_value
            # Format the time in the desired format
            end_time = current_time.isoformat()
            # Define the endpoint for market candles, including query parameters
            endpoint = f"v1/markets/candles/{symbol_id}?startTime={start_time}&endTime={end_time}&interval={interval}"

            # Make the API call
            response = make_api_call(access_token, server_url, endpoint)

            if response.status_code == 200:
                return response.json()  # Return the successful response
            else:
                print("Error:", response.status_code)
                return None  # Return None or an appropriate error message
        else:
            print("Access token or server URL is missing.")
            return None
    else:
        print("Could not retrieve API key data.")
        return None

def add_json_to_csv(json_data, csv_filename):
    # Define the directory path
    directory_path = 'data/candles/'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Full path for the CSV file
    full_csv_path = os.path.join(directory_path, csv_filename)

    # Load existing CSV file into a DataFrame
    try:
        existing_df = pd.read_csv(full_csv_path)
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    # Convert JSON data to DataFrame
    new_df = pd.json_normalize(json_data['candles'])

    # Check if new_df is empty or if columns don't match
    if new_df.empty or not set(existing_df.columns).issubset(set(new_df.columns)):
        print(f"No data to add for {csv_filename}")
        return

    # Ensure that the DataFrame has the same column order
    new_df = new_df[existing_df.columns]

    # Remove duplicates
    combined_df = pd.concat([existing_df, new_df]).drop_duplicates()

    # Save the updated DataFrame to CSV
    combined_df.to_csv(full_csv_path, index=False)
    # Print confirmation message
    print(f"New data added to {csv_filename}")
    

# Rate limiting parameters
MAX_REQUESTS_PER_SECOND = 20
MAX_REQUESTS_PER_HOUR = 15000
lock = Lock()
requests_count = 0
start_time = time.time()

def rate_limited_api_call(symbol_id, interval, end_value):
    global requests_count
    with lock:
        # Check if hour limit is reached
        if requests_count >= MAX_REQUESTS_PER_HOUR:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time < 3600:  # Less than an hour
                time.sleep(3600 - elapsed_time)  # Sleep until the hour resets
            # Reset counters
            requests_count = 0
            start_time = time.time()
        else:
            # Check for second limit
            time.sleep(1 / MAX_REQUESTS_PER_SECOND)

        requests_count += 1

    return get_market_candles(symbol_id, interval, end_value)

def process_csv_files(directory_path):
    def process_file(filename):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)

        info = extract_symbol_info(filename)
        if info:
            symbol, interval, end_value, symbol_description, symbol_id, security_type, listing_exchange, is_tradable, is_quotable, currency = info
            json_data = rate_limited_api_call(symbol_id, interval, end_value)
            if json_data:
                print(f'Processing data for {symbol} {symbol_description}')
                add_json_to_csv(json_data, filename)

    with ThreadPoolExecutor() as executor:
        futures = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                futures.append(executor.submit(process_file, filename))

        for future in futures:
            future.result()

process_csv_files("data/candles")


