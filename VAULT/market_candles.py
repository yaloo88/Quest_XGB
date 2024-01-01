import requests
import datetime
from api_auth import get_api_key_data, make_api_call
from symbol_search import get_symbols_search

def calculate_start_time(end_time, interval, max_candles=2000):
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

# Example usage of the function
symbol_id = input("Enter the symbol ID: ")
interval = input("Enter the interval (e.g., OneMinute, OneHour, OneDay, etc.): ")
get_market_candles(symbol_id, interval)