from matplotlib import pyplot as plt
import pandas as pd
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score
import requests
from math import ceil
from joblib import Parallel, delayed

interval = 'OneMinute' # Replace with your candle interval
webhook_url = '' # Replace with your webhook URL

# Create a dictionary to store cached data
cached_data = {}

def read_stock_data(symbol):
    '''
    Command to use:
    symbol = input("Enter the stock symbol: ")
    data = read_stock_data(symbol)
    print(data)
    '''
    global cached_data
    
    # Check if the data is already cached
    if symbol in cached_data:
        return cached_data[symbol]

    # Construct the file path based on the symbol
    file_path = f"data/candles/{symbol}_{interval}.csv"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    # Reading the CSV file in chunks
    chunk_size = 10000  # Adjust as needed
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    
    # Concatenate chunks into a single DataFrame
    data = pd.concat(chunks, ignore_index=True)
    
    # Converting 'start' and 'end' columns to date/time format
    data['start'] = pd.to_datetime(data['start'], utc=True)
    data['end'] = pd.to_datetime(data['end'], utc=True)  # Convert to UTC

    # Setting 'end' column as the index and converting it to DatetimeIndex
    data.set_index('end', inplace=True)
    
    # Cache the data for future use
    cached_data[symbol] = data
    
    return data

def process_candles_data(symbol_candles):
    '''
    Command to use:
    hourly_candles = process_candles_data(data)
    print(hourly_candles)
    '''
    # Resample to hourly intervals
    hourly_candles = symbol_candles.resample('1H').agg({
        'low': 'min',
        'high': 'max',
        'open': 'first',
        'close': 'last',
        'volume': 'sum',
        'VWAP': 'mean'
    }).dropna()

    # Parameters for various indicators
    rolling_window = 20
    ema_short_span = 12
    ema_long_span = 26
    rsi_window = 14
    macd_signal_span = 9

    # Calculate SMA and EMA
    hourly_candles['SMA_10'] = hourly_candles['close'].rolling(window=10).mean()
    hourly_candles['SMA_30'] = hourly_candles['close'].rolling(window=30).mean()
    hourly_candles['SMA_50'] = hourly_candles['close'].rolling(window=50).mean()
    hourly_candles['SMA_100'] = hourly_candles['close'].rolling(window=100).mean()
    hourly_candles['SMA_200'] = hourly_candles['close'].rolling(window=200).mean()
    hourly_candles['EMA_50'] = hourly_candles['close'].ewm(span=50, adjust=False).mean()
    hourly_candles['EMA_100'] = hourly_candles['close'].ewm(span=100, adjust=False).mean()
    hourly_candles['EMA_200'] = hourly_candles['close'].ewm(span=200, adjust=False).mean()
    hourly_candles['EMA_10'] = hourly_candles['close'].ewm(span=10, adjust=False).mean()
    hourly_candles['EMA_30'] = hourly_candles['close'].ewm(span=30, adjust=False).mean()

    # Calculate Bollinger Bands
    rolling_std = hourly_candles['close'].rolling(window=rolling_window).std()
    hourly_candles['middle_BB'] = hourly_candles['close'].rolling(window=rolling_window).mean()
    hourly_candles['upper_BB'] = hourly_candles['middle_BB'] + 2 * rolling_std
    hourly_candles['lower_BB'] = hourly_candles['middle_BB'] - 2 * rolling_std

    # Calculate RSI
    delta = hourly_candles['close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    hourly_candles['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    ema_short = hourly_candles['close'].ewm(span=ema_short_span, adjust=False).mean()
    ema_long = hourly_candles['close'].ewm(span=ema_long_span, adjust=False).mean()
    hourly_candles['MACD'] = ema_short - ema_long
    hourly_candles['MACD_signal'] = hourly_candles['MACD'].ewm(span=macd_signal_span, adjust=False).mean()
    hourly_candles['MACD_hist'] = hourly_candles['MACD'] - hourly_candles['MACD_signal']

    # Dropping the rows with NaN values
    hourly_candles.dropna(inplace=True)

    # Defining the target variables by shifting the high and low prices by one hour into the future
    hourly_candles['next_hour_high'] = hourly_candles['high'].shift(-1)
    hourly_candles['next_hour_low'] = hourly_candles['low'].shift(-1)

    return hourly_candles

def prepare_train_test_data(hourly_candles):
    '''
    # Example usage:
    (X_train_high, X_test_high, y_train_high, y_test_high), (X_train_low, X_test_low, y_train_low, y_test_low) = prepare_train_test_data(hourly_candles)
    '''
    # Dropping the rows with NaN values for 'next_hour_high' and 'next_hour_low'
    hourly_candles.dropna(subset=['next_hour_high', 'next_hour_low'], inplace=True)

    # Separating the features and target variables
    features = hourly_candles.drop(columns=['next_hour_high', 'next_hour_low'])
    targets_high = hourly_candles['next_hour_high']
    targets_low = hourly_candles['next_hour_low']

    # Splitting the data into training and testing sets (80% training, 20% testing)
    train_size = int(0.85 * len(hourly_candles))
    X_train, X_test = features[:train_size], features[train_size:]
    y_train_high, y_test_high = targets_high[:train_size], targets_high[train_size:]
    y_train_low, y_test_low = targets_low[:train_size], targets_low[train_size:]

    return (X_train, X_test, y_train_high, y_test_high), (X_train, X_test, y_train_low, y_test_low)

def train_evaluate_xgboost_model(X_train, y_train, X_test, y_test):
    '''
    Command to use:
    # Train and evaluate the XGBoost model for high prices
    model_high, mae_high, rmse_high, avg_mae_cv_high = train_evaluate_xgboost_model(X_train_high, y_train_high, X_test_high, y_test_high)

    # Train and evaluate the XGBoost model for low prices
    model_low, mae_low, rmse_low, avg_mae_cv_low = train_evaluate_xgboost_model(X_train_low, y_train_low, X_test_low, y_test_low)

    '''
    model = XGBRegressor(n_estimators=200, objective='reg:squarederror', n_jobs=-1, random_state=42)

    # Perform cross-validation on the training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    avg_mae_cv = -np.mean(cv_scores)

    # Train the model on the full training set
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    return model, mae, rmse, avg_mae_cv

def train_predict_plot(X_train_high, y_train_high, X_train_low, y_train_low, X_test_high, y_test_high, X_test_low, y_test_low, title, symbol):
    '''
    print('Processed:', symbol)
    train_predict_plot(X_train_high, y_train_high, X_train_low, y_train_low, X_test_high, y_test_high, X_test_low, y_test_low, f"Next Hour's High and Low Price for {symbol}: Actual vs. Predicted", symbol)
    print('----------------------------------------------')
    '''
    # Initialize the XGBoost Regressor for high prices with parallelization
    model_high, _, _, _ = train_evaluate_xgboost_model(X_train_high, y_train_high, X_test_high, y_test_high)
    predictions_high = model_high.predict(X_test_high)

    # Initialize the XGBoost Regressor for low prices with parallelization
    model_low, _, _, _ = train_evaluate_xgboost_model(X_train_low, y_train_low, X_test_low, y_test_low)
    predictions_low = model_low.predict(X_test_low)

    plt.figure(figsize=(15, 8))

    # Plot and fill the area between the actual high and low values
    plt.plot(y_test_high.index, y_test_high, color='blue')
    plt.plot(y_test_low.index, y_test_low, color='blue')
    plt.fill_between(y_test_high.index, y_test_high, y_test_low, facecolor='blue', alpha=0.2, label="Actual Band")

    # Plot and fill the area between the predicted high and low values
    plt.plot(y_test_high.index, predictions_high, linestyle='dashed', color='red')
    plt.plot(y_test_low.index, predictions_low, linestyle='dashed', color='red')
    plt.fill_between(y_test_high.index, predictions_high, predictions_low, facecolor='red', alpha=0.2, label="Predicted Band")

    # Check if the actual value is 5% lower than the predicted value for high prices
    flags_high = (y_test_high > predictions_high * 1.01)
    flagged_indices_high = y_test_high.index[flags_high]

    # Check if the actual value is 5% lower than the predicted value for low prices
    flags_low = (y_test_low < predictions_low * 0.99)
    flagged_indices_low = y_test_low.index[flags_low]

    # Plot flagged points for high prices
    plt.scatter(flagged_indices_high, y_test_high[flags_high], color='red', label='Flagged High')

    # Plot flagged points for low prices
    plt.scatter(flagged_indices_low, y_test_low[flags_low], color='green', label='Flagged Low')

    # Add horizontal lines at y-tick locations
    y_ticks = plt.yticks()[0]
    for level in y_ticks:
        plt.axhline(y=level, color='gray', linestyle='--', linewidth=0.5)

    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"data/graphs/{symbol}_graph.png")
    #plt.show() # Un-comment to show the graph
    
    # Send Discord embed message if condition is met for the last data point
    if flags_low.iloc[-1]:
        send_discord_alert(symbol)
    
    return model_high, predictions_high, model_low, predictions_low, flagged_indices_high, flagged_indices_low

def send_discord_alert(symbol):

    # Embed payload without the image
    embed = {
        "title": f"Alert for {symbol}",
        "description": f"The actual low value is less than 99% of the predicted value for {symbol}.",
        "color": 16711680 # Red color
    }
    payload = {
        "embeds": [embed]
    }

    # Open the image as a binary file
    with open(f"data/graphs/{symbol}_graph.png", "rb") as image_file:
        # Send the POST request with both the embed and the file attached
        response = requests.post(
            webhook_url,
            json=payload,
            files={"file": ("image.png", image_file, "image/png")} # Attach the image file
        )

    if response.status_code != 204:
        print(f"Failed to send Discord message: {response.text}")





# Get the list of all CSV files in the specified directory
symbol_files = [f for f in os.listdir("data/candles") 
                if os.path.isfile(os.path.join("data/candles", f)) and f.endswith(f"_{interval}.csv")]

# Extract the symbols from the file names
symbols = [f.split(f"_{interval}.csv")[0] for f in symbol_files]

# Define the batch size
batch_size = 5  # You can modify this as per your requirement

# Calculate the number of batches
num_batches = ceil(len(symbols) / batch_size)


def process_batch(batch_symbols):
    for symbol in batch_symbols:
        print(f"Processing symbol: {symbol}")
        
        # Read stock data
        data = read_stock_data(symbol)

        # Process candles data
        hourly_candles = process_candles_data(data)

        # Prepare train-test data
        (X_train_high, X_test_high, y_train_high, y_test_high), (X_train_low, X_test_low, y_train_low, y_test_low) = prepare_train_test_data(hourly_candles)

        # Train and evaluate the XGBoost model for high prices
        model_high, mae_high, rmse_high, avg_mae_cv_high = train_evaluate_xgboost_model(X_train_high, y_train_high, X_test_high, y_test_high)

        # Train and evaluate the XGBoost model for low prices
        model_low, mae_low, rmse_low, avg_mae_cv_low = train_evaluate_xgboost_model(X_train_low, y_train_low, X_test_low, y_test_low)

        print('Processed:', symbol)
        train_predict_plot(X_train_high, y_train_high, X_train_low, y_train_low, X_test_high, y_test_high, X_test_low, y_test_low, f"{symbol}: Actual vs. Predicted", symbol)
        print('----------------------------------------------')





# This function will be called in parallel for each batch of symbols
def process_batch_parallel(batch_symbols):
    for symbol in batch_symbols:
        print(f"Processing symbol: {symbol}")
        
        # Read stock data
        data = read_stock_data(symbol)

        # Process candles data
        hourly_candles = process_candles_data(data)

        # Prepare train-test data
        (X_train_high, X_test_high, y_train_high, y_test_high), (X_train_low, X_test_low, y_train_low, y_test_low) = prepare_train_test_data(hourly_candles)

        # Train and evaluate the XGBoost model for high prices
        model_high, mae_high, rmse_high, avg_mae_cv_high = train_evaluate_xgboost_model(X_train_high, y_train_high, X_test_high, y_test_high)

        # Train and evaluate the XGBoost model for low prices
        model_low, mae_low, rmse_low, avg_mae_cv_low = train_evaluate_xgboost_model(X_train_low, y_train_low, X_test_low, y_test_low)

        print('Processed:', symbol)
        train_predict_plot(X_train_high, y_train_high, X_train_low, y_train_low, X_test_high, y_test_high, X_test_low, y_test_low, f"{symbol}: Actual vs. Predicted", symbol)
        print('----------------------------------------------')

# Define the number of parallel jobs (usually equal to the number of CPU cores)
n_jobs = -1  # This will utilize all available CPU cores

# Iterate through batches and call the process_batch_parallel function in parallel
Parallel(n_jobs=n_jobs)(delayed(process_batch_parallel)(symbols[i * batch_size: (i + 1) * batch_size]) for i in range(num_batches))
