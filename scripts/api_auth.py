import requests
import json
import os
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

api_key_data = {}

# Function to make an API call
def make_api_call(access_token, server_url, endpoint):
    """
    Makes an authorized API call to the specified endpoint.

    :param access_token: The access token for API authorization.
    :param server_url: The base URL of the API server.
    :param endpoint: The API endpoint to call.
    :return: The response from the API call.
    """

    # Ensure there's exactly one slash between the server URL and the endpoint
    url = f"{server_url.rstrip('/')}/{endpoint.lstrip('/')}"

    # Set up the headers with the authorization token
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Return the response
    return response

# Function to generate or load the encryption key
def get_encryption_key():
    key_file_path = 'secrets/encryption_key.key'
    
    if os.path.exists(key_file_path):
        with open(key_file_path, 'rb') as key_file:
            encryption_key = key_file.read()
    else:
        # Generate a new encryption key
        encryption_key = Fernet.generate_key()
        
        # Save the encryption key to a file for future use
        with open(key_file_path, 'wb') as key_file:
            key_file.write(encryption_key)
    
    return encryption_key

# Function to get the API key data
def get_api_key_data():
    '''
    # If you already have the API key data, you can load it from a file
    api_key_data = get_api_key_data()
    '''
    encryption_key = get_encryption_key()
    
    # Check if api_key.json file exists
    if os.path.exists('secrets/api_key.json'):
        with open('secrets/api_key.json', 'rb') as json_file:
            encrypted_data = json_file.read()
        
        fernet = Fernet(encryption_key)
        
        # Decrypt the data
        decrypted_data = fernet.decrypt(encrypted_data)
        
        # Deserialize the JSON data
        api_key_data = json.loads(decrypted_data.decode())
    else:
        # If it doesn't exist, prompt the user for a key
        print("-" * 40)
        print("You need to get an API key from Questrade.")
        print("Go to https://www.questrade.com/api/documentation/getting-started")          
        refresh_token = input("Enter your refresh token: ")
        response = requests.get(f"https://login.questrade.com/oauth2/token?grant_type=refresh_token&refresh_token={refresh_token}")

        if response.status_code == 200:
            api_key_data = response.json()
            
            # Serialize the API key data to JSON
            api_key_json = json.dumps(api_key_data)
            
            fernet = Fernet(encryption_key)
            
            # Encrypt the data
            encrypted_data = fernet.encrypt(api_key_json.encode())
            
            # Specify the file path where you want to save the encrypted JSON output
            output_file_path = 'secrets/api_key.json'
            
            # Write the encrypted data to the file
            with open(output_file_path, 'wb') as json_file:
                json_file.write(encrypted_data)
            
            print(f"Response saved to {output_file_path}")
        else:
            print(f"Error: Received a non-200 status code ({response.status_code})")
            return None

    return api_key_data

# Function to refresh the token
def refresh_token(refresh_token):
    '''
    # You can refresh the token
    api_key_data = get_api_key_data()
    if api_key_data:
    refresh_token(api_key_data['refresh_token'])
    '''
    encryption_key = get_encryption_key()
    
    response = requests.post(
        f"https://login.questrade.com/oauth2/token",
        data={
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token
        }
    )
    
    if response.status_code == 200:
        new_api_key_data = response.json()
        
        api_key_data.update(new_api_key_data)
        
        fernet = Fernet(encryption_key)
        
        # Serialize the updated API key data to JSON
        updated_api_key_json = json.dumps(api_key_data)
        
        # Encrypt the updated data
        encrypted_data = fernet.encrypt(updated_api_key_json.encode())
        
        # Save the updated API key data to api_key.json
        with open('secrets/api_key.json', 'wb') as json_file:
            json_file.write(encrypted_data)
        
        print("Token refreshed successfully.")
    else:
        print(f"Error: Received a non-200 status code ({response.status_code})")

api_key_data = get_api_key_data()
if api_key_data:
    refresh_token(api_key_data['refresh_token'])