import csv
import json
import os
import requests
import datetime
from api_auth import get_api_key_data, make_api_call, refresh_token

def calculate_start_time(end_time, interval, max_candles=100000):
    """
    Calculate the start time based on end time, interval, and maximum number of candles.
    """
    interval_durations = {
        "OneMinute": datetime.timedelta(minutes=1),
        "TwoMinutes": datetime.timedelta(minutes=2),
        "ThreeMinutes": datetime.timedelta(minutes=3),
        "FourMinutes": datetime.timedelta(minutes=4),
        "FiveMinutes": datetime.timedelta(minutes=5),
        "TenMinutes": datetime.timedelta(minutes=10),
        "FifteenMinutes": datetime.timedelta(minutes=15),
        "TwentyMinutes": datetime.timedelta(minutes=20),
        "HalfHour": datetime.timedelta(minutes=30),
        "OneHour": datetime.timedelta(hours=1),
        "TwoHours": datetime.timedelta(hours=2),
        "FourHours": datetime.timedelta(hours=4),
        "OneDay": datetime.timedelta(days=1),
        "OneWeek": datetime.timedelta(weeks=1),
        "OneMonth": datetime.timedelta(days=30),  # Approximation
        "OneYear": datetime.timedelta(days=365)   # Approximation
    }

    total_duration = interval_durations[interval] * max_candles
    start_time = end_time - total_duration
    return start_time

def get_market_candles(symbol_id, interval):
    # Retrieve API key data
    api_key_data = get_api_key_data()
    if api_key_data:
        # Accessing the access_token and server URL from the dictionary
        access_token = api_key_data.get('access_token')
        server_url = api_key_data.get('api_server')

        if access_token and server_url:
            # Calculate start and end times
            end_time = datetime.datetime.now(datetime.timezone.utc)
            start_time = calculate_start_time(end_time, interval)

            # Define the endpoint for market candles, including query parameters
            endpoint = f"v1/markets/candles/{symbol_id}?startTime={start_time.isoformat()}&endTime={end_time.isoformat()}&interval={interval}"

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

def json_to_csv(json_data, filename):
    # Extract the list of candles from the JSON data
    candle_data = json_data.get('candles', [])
    
    if candle_data and isinstance(candle_data, list) and len(candle_data) > 0 and isinstance(candle_data[0], dict):
        with open(filename, 'w', newline='') as file:
            fieldnames = candle_data[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            for row in candle_data:
                writer.writerow(row)
        print(f"CSV file saved as '{filename}'")
    else:
        print("Invalid JSON data: No CSV file created")
        print(json_data)  # Debug: Print the JSON data for inspection

def get_all_symbol_ids(directory):
    """
    Reads all JSON files in the given directory and extracts symbol IDs along with their symbols.
    """
    symbol_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                symbol_id = data.get('symbolId')
                symbol = data.get('symbol')
                if symbol_id is not None and symbol is not None:
                    symbol_data.append((symbol, symbol_id))
    return symbol_data

def file_exists(symbol, interval, path):
    """
    Checks if a file for a given symbol and interval already exists.
    """
    filename = f"{path}\\{symbol}_{interval}.csv"
    return os.path.isfile(filename)

def download_all_market_data(symbol_data, interval, path):
    """
    Downloads market data for all given symbol IDs and saves them as CSV files.
    Only downloads data that doesn't already exist.
    """
    for symbol, symbol_id in symbol_data:
        if not file_exists(symbol, interval, path):
            candles = get_market_candles(symbol_id, interval)
            if candles:
                filename = f"{path}\\{symbol}_{interval}.csv"
                json_to_csv(candles, filename)
        else:
            print(f"Data already exists for {symbol} at interval {interval}, skipping download.")

# Example usage
symbol_data = get_all_symbol_ids("data\symbols")
interval = input("Enter the interval (e.g., OneMinute, OneHour, OneDay, etc.): ")
path = "data\candles"
download_all_market_data(symbol_data, interval, path)