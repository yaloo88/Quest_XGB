import requests
import csv
from api_auth import get_api_key_data, make_api_call

def get_accounts():
    '''
    # Example usage of the modified function
    print(get_accounts())
    '''
    # Retrieve API key data
    api_key_data = get_api_key_data()
    if api_key_data:
        # Accessing the access_token and server URL from the dictionary
        access_token = api_key_data.get('access_token')
        server_url = api_key_data.get('api_server')

        if access_token and server_url:
            # Define the endpoint for accounts
            endpoint = "v1/accounts"
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

def get_account_positions(access_token, server_url, account_id):
    # Define the endpoint for account positions, including the account ID
    endpoint = f"v1/accounts/{account_id}/positions"
    response = make_api_call(access_token, server_url, endpoint)
    
    if response.status_code == 200:
        return response.json()  # Return the successful response
    else:
        print("Error:", response.status_code)
        return None  # Return None or an appropriate error message

def get_all_accounts_positions():
    '''
    # Example usage
    positions_data = get_all_accounts_positions()
    '''
    # Retrieve API key data
    api_key_data = get_api_key_data()
    all_positions = []

    if api_key_data:
        access_token = api_key_data.get('access_token')
        server_url = api_key_data.get('api_server')

        if access_token and server_url:
            accounts = get_accounts()
            if accounts:
                for account in accounts.get('accounts', []):
                    account_id = account.get('number')
                    if account_id:
                        positions = get_account_positions(access_token, server_url, account_id)
                        all_positions.append((account_id, positions))
            else:
                print("No accounts retrieved.")
        else:
            print("Access token or server URL is missing.")
    else:
        print("Could not retrieve API key data.")

    return all_positions

def extract_symbols_and_save_to_csv(positions_data, csv_file_path):
    unique_symbols = set()

    for _, positions in positions_data:
        if positions and 'positions' in positions:
            symbols = [position.get('symbol') for position in positions['positions'] if position.get('symbol')]
            unique_symbols.update(symbols)
            
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(unique_symbols))


positions_data = get_all_accounts_positions()
csv_file_path = 'market_list\PORTFOLIO.csv'
extract_symbols_and_save_to_csv(positions_data, csv_file_path)

print("-" * 40)
print("Portfolio symbols list saved to:", csv_file_path)
print('Please open the file and copy the content to the clipboard.')
print("-" * 40)

