from api_auth import get_api_key_data, make_api_call


def get_symbols_search():
    # Retrieve API key data
    api_key_data = get_api_key_data()
    if not api_key_data:
        print("Could not retrieve API key data.")
        return

    # Accessing the access_token and server URL from the dictionary
    access_token = api_key_data.get('access_token')
    server_url = api_key_data.get('api_server')

    if not access_token or not server_url:
        print("Access token or server URL is missing.")
        return

    # Prompt user for the symbol prefix
    symbol_prefix = input("Enter the symbol prefix to search: ").upper()

    # Modify the endpoint to include the query for searching symbols with the user-defined prefix
    endpoint = f"v1/symbols/search?prefix={symbol_prefix}"
    response = make_api_call(access_token, server_url, endpoint)

    if response.status_code == 200:
        data = response.json()
        print("Success:", data)

    else:
        print("Error:", response.status_code)

# Example usage of the function
get_symbols_search()

