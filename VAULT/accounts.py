import requests

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
    get_all_accounts_positions()
    '''
    # Retrieve API key data
    api_key_data = get_api_key_data()
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
                        print(f"Positions for Account {account_id}: {positions}")
                    else:
                        print("No account number found for the account.")
            else:
                print("No accounts retrieved.")
        else:
            print("Access token or server URL is missing.")
    else:
        print("Could not retrieve API key data.")



# Example usage
positions = get_all_accounts_positions()
print(positions)