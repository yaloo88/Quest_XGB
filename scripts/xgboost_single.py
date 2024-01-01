from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from market_candles import get_market_candles, symbol_id, interval
import pandas as pd
import json
from xgboost import XGBRegressor

###############################################################################
# Load the data
###############################################################################
# Get market candles data and convert to a dictionary
json_data = get_market_candles(symbol_id, interval)
data = json.loads(json_data) if isinstance(json_data, str) else json_data


def calculate_technical_indicators(data):
    data = pd.json_normalize(data['candles'])
    data['start'] = pd.to_datetime(data['start'], utc=True)
    # Drop the 'end' column from the DataFrame
    data.drop(columns=['end'], inplace=True)

    # Parameters for various indicators
    ema_params = [10, 30, 50, 100, 200]
    sma_params = [10, 30, 50, 100, 200]
    rolling_window = 20
    ema_short_span = 12
    ema_long_span = 26
    rsi_window = 14
    macd_signal_span = 9

    # Calculate SMA and EMA using loops
    for param in sma_params:
        data[f'SMA_{param}'] = data['close'].rolling(window=param).mean()

    for param in ema_params:
        data[f'EMA_{param}'] = data['close'].ewm(span=param, adjust=False).mean()

    # Calculate Bollinger Bands
    rolling_std = data['close'].rolling(window=rolling_window).std()
    data['middle_BB'] = data['close'].rolling(window=rolling_window).mean()
    data['upper_BB'] = data['middle_BB'] + 2 * rolling_std
    data['lower_BB'] = data['middle_BB'] - 2 * rolling_std

    # Calculate RSI
    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    ema_short = data['close'].ewm(span=ema_short_span, adjust=False).mean()
    ema_long = data['close'].ewm(span=ema_long_span, adjust=False).mean()
    data['MACD'] = ema_short - ema_long
    data['MACD_signal'] = data['MACD'].ewm(span=macd_signal_span, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']

    # Clean data by dropping NaN values
    data.dropna(inplace=True)

    # Defining target variables by shifting future prices
    data['next_high'] = data['high'].shift(-1)
    data['next_low'] = data['low'].shift(-1)

    # Set 'start' column as the index
    data.set_index('start', inplace=True)

    return data

def prepare_train_test_data(data):
    '''
    # Example usage:
    (X_train_high, X_test_high, y_train_high, y_test_high), (X_train_low, X_test_low, y_train_low, y_test_low) = prepare_train_test_data(data)
    '''
    # Dropping the rows with NaN values for 'next_high' and 'next_low'
    data.dropna(subset=['next_high', 'next_low'], inplace=True)

    # Separating the features and target variables
    features = data.drop(columns=['next_high', 'next_low'])
    targets_high = data['next_high']
    targets_low = data['next_low']

    # Splitting the data into training and testing sets (80% training, 20% testing)
    train_size = int(0.85 * len(data))
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

def train_predict_plot(X_train_high, y_train_high, X_train_low, y_train_low, 
                       X_test_high, y_test_high, X_test_low, y_test_low 
                       ):
    # Assuming train_evaluate_xgboost_model and send_discord_alert are defined elsewhere

    # Train and predict for high prices
    model_high, _, _, _ = train_evaluate_xgboost_model(X_train_high, y_train_high, X_test_high, y_test_high)
    predictions_high = model_high.predict(X_test_high)

    # Train and predict for low prices
    model_low, _, _, _ = train_evaluate_xgboost_model(X_train_low, y_train_low, X_test_low, y_test_low)
    predictions_low = model_low.predict(X_test_low)

    # Creating the plot
    plt.figure(figsize=(15, 8))

    # Plot actual high and low prices
    plt.plot(y_test_high.index, y_test_high, color='blue', label='Actual Band')
    plt.plot(y_test_low.index, y_test_low, color='blue')
    plt.fill_between(y_test_high.index, y_test_high, y_test_low, facecolor='blue', alpha=0.2)

    # Plot predicted high and low prices
    plt.plot(y_test_high.index, predictions_high, linestyle='dashed', color='red', label='Predicted Band')
    plt.plot(y_test_low.index, predictions_low, linestyle='dashed', color='red')
    plt.fill_between(y_test_high.index, predictions_high, predictions_low, facecolor='red', alpha=0.2)

    # Flagging logic for anomalies
    flags_high = (y_test_high > predictions_high * 1.01)
    flagged_indices_high = y_test_high.index[flags_high]

    flags_low = (y_test_low < predictions_low * 0.99)
    flagged_indices_low = y_test_low.index[flags_low]

    # Plot flagged points
    plt.scatter(flagged_indices_high, y_test_high[flags_high], color='red', label='Sell')
    plt.scatter(flagged_indices_low, y_test_low[flags_low], color='green', label='Buy')

    # Plot enhancements
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


    return model_high, predictions_high, model_low, predictions_low, flagged_indices_high, flagged_indices_low

# Example usage:
data = calculate_technical_indicators(data)
print(data.head())
print('WORKING..Please be patient..')
(X_train_high, X_test_high, y_train_high, y_test_high), (X_train_low, X_test_low, y_train_low, y_test_low) = prepare_train_test_data(data)
# Train and evaluate the XGBoost model for high prices
model_high, mae_high, rmse_high, avg_mae_cv_high = train_evaluate_xgboost_model(X_train_high, y_train_high, X_test_high, y_test_high)
# Train and evaluate the XGBoost model for low prices
model_low, mae_low, rmse_low, avg_mae_cv_low = train_evaluate_xgboost_model(X_train_low, y_train_low, X_test_low, y_test_low)

train_predict_plot(X_train_high, y_train_high, X_train_low, y_train_low, 
                       X_test_high, y_test_high, X_test_low, y_test_low
                       )