import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

def prepare_data_for_linear_regression(data):
    """Prepare data for linear regression"""
    # Select features
    X = data[['ElapsedDays', 'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']].copy()
    y = data['Close'].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_linear_regression_model(data):
    """Train Linear Regression model"""
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data_for_linear_regression(data)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    results = {
        'model': model,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test,
        'metrics': {
            'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
            'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
        }
    }
    
    return results

def prepare_data_for_arima(data):
    """Prepare data for ARIMA model"""
    # Use Close prices as time series
    stock_ts = data.set_index('Date')[['Close']].copy()
    stock_ts = stock_ts.sort_index()
    
    # Split data (80% train, 20% test)
    split_point = int(len(stock_ts) * 0.8)
    train_ts = stock_ts[:split_point]
    test_ts = stock_ts[split_point:]
    
    return train_ts, test_ts, stock_ts

def train_arima_model(data, order=(2, 1, 2)):
    """Train ARIMA model"""
    if not ARIMA_AVAILABLE:
        raise ImportError("statsmodels is not installed. Please install it to use ARIMA.")
    
    # Prepare data
    train_ts, test_ts, stock_ts = prepare_data_for_arima(data)
    
    try:
        # Fit model
        model = ARIMA(train_ts['Close'], order=order)
        fitted_model = model.fit()
        
        # Make predictions
        predictions = fitted_model.forecast(steps=len(test_ts))
        
        # Calculate metrics
        test_mae = mean_absolute_error(test_ts['Close'], predictions)
        test_rmse = np.sqrt(mean_squared_error(test_ts['Close'], predictions))
        test_r2 = r2_score(test_ts['Close'], predictions)
        
        results = {
            'model': fitted_model,
            'predictions': predictions,
            'test_actual': test_ts['Close'].values,
            'train_data': train_ts,
            'test_data': test_ts,
            'metrics': {
                'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
            }
        }
        
        return results
    except Exception as e:
        raise Exception(f"Error training ARIMA model: {str(e)}")

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def prepare_data_for_lstm(data):
    """Prepare data for LSTM model"""
    # Use Close prices
    stock_ts = data.set_index('Date')[['Close']].copy()
    stock_ts = stock_ts.sort_index()
    
    # Split data
    split_point = int(len(stock_ts) * 0.8)
    train_ts = stock_ts[:split_point]
    test_ts = stock_ts[split_point:]
    
    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_ts[['Close']])
    test_scaled = scaler.transform(test_ts[['Close']])
    
    return train_scaled, test_scaled, scaler, train_ts, test_ts

def train_lstm_model(data, seq_length=60, epochs=50, batch_size=32):
    """Train LSTM model"""
    if not LSTM_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Please install it to use LSTM.")
    
    # Prepare data
    train_scaled, test_scaled, scaler, train_ts, test_ts = prepare_data_for_lstm(data)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled.flatten(), seq_length)
    X_test, y_test = create_sequences(test_scaled.flatten(), seq_length)
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Make predictions
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train_actual, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
    train_r2 = r2_score(y_train_actual, train_pred)
    
    test_mae = mean_absolute_error(y_test_actual, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
    test_r2 = r2_score(y_test_actual, test_pred)
    
    results = {
        'model': model,
        'scaler': scaler,
        'train_predictions': train_pred,
        'test_predictions': test_pred,
        'y_train_actual': y_train_actual,
        'y_test_actual': y_test_actual,
        'history': history.history,
        'metrics': {
            'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
            'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
        }
    }
    
    return results

