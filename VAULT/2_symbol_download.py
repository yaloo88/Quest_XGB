import os
import json
from api_auth import get_api_key_data, make_api_call

def get_symbols_download(symbol_prefix):
    # Accessing the access_token and server URL from the dictionary
    api_key_data = get_api_key_data()
    if not api_key_data:
        print("Could not retrieve API key data.")
        return None

    access_token = api_key_data.get('access_token')
    server_url = api_key_data.get('api_server')

    if not access_token or not server_url:
        print("Access token or server URL is missing.")
        return None

    # Modify the endpoint to include the query for searching symbols with the user-defined prefix
    endpoint = f"v1/symbols/search?prefix={symbol_prefix}"
    response = make_api_call(access_token, server_url, endpoint)

    if response.status_code == 200:
        data = response.json()
        # Extract the list of symbols from the response data
        symbols = data.get('symbols', [])
        if symbols and isinstance(symbols, list):
            return symbols[0]
    else:
        print("Error for symbol", symbol_prefix, ":", response.status_code)
    return None

def process_symbols_input():
    directory = 'data\\symbols'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Input from the user
    symbols_input = input("Enter the symbols separated by commas: ")
    symbols = symbols_input.split(',')

    for symbol in symbols:
        first_result = get_symbols_download(symbol.strip())
        if first_result:
            # Save the results to a file
            file_path = os.path.join(directory, f'{symbol.strip()}.json')
            with open(file_path, 'w') as file:
                json.dump(first_result, file, indent=4)
            print(f"First result for {symbol.strip()} saved to file: {file_path}")

process_symbols_input()
