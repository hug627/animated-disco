import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils import (
    train_linear_regression_model,
    train_arima_model,
    train_lstm_model,
    ARIMA_AVAILABLE,
    LSTM_AVAILABLE
)

# Page configuration
st.set_page_config(
    page_title="Amazon Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #232F3E;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9900;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'lr_model' not in st.session_state:
    st.session_state.lr_model = None
if 'arima_model' not in st.session_state:
    st.session_state.arima_model = None
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {'lr': False, 'arima': False, 'lstm': False}

# Sidebar navigation
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Choose a page",
    ["🏠 Home", "📊 Data Overview", "📈 Visualizations", "🤖 Model Training", "🔮 Predictions", "📉 Model Comparison"]
)

# Load data function
@st.cache_data
def load_data(file_path_or_uploaded):
    """Load and preprocess stock data"""
    try:
        # Handle both file paths and uploaded files
        if isinstance(file_path_or_uploaded, str):
            df = pd.read_csv(file_path_or_uploaded)
        else:
            # It's an uploaded file object
            df = pd.read_csv(file_path_or_uploaded)
        
        # Debug: Show column names and first few rows
        st.write("**Debug Information:**")
        st.write("📋 Columns found:", df.columns.tolist())
        st.write("📊 First 5 rows:")
        st.dataframe(df.head())
        
        # Find the date column (case-insensitive)
        date_column = None
        for col in df.columns:
            if col.lower().strip() in ['date', 'datetime', 'timestamp', 'time']:
                date_column = col
                break
        
        if date_column is None:
            st.error("❌ No date column found! Please ensure your CSV has a column named 'Date'")
            st.info("Available columns: " + ", ".join(df.columns.tolist()))
            return None
        
        st.write(f"📅 Using date column: '{date_column}'")
        st.write(f"📝 Sample date values: {df[date_column].head(3).tolist()}")
        st.write(f"📦 Original date column type: {df[date_column].dtype}")
        
        # Convert to datetime - try multiple approaches
        st.write("🔄 Attempting date conversion...")
        
        # Method 1: Try with infer_datetime_format (faster)
        try:
            df['Date'] = pd.to_datetime(df[date_column], infer_datetime_format=True, errors='coerce')
            st.write(f"Method 1 type: {df['Date'].dtype}")
        except:
            pass
        
        # If still not datetime, try without infer
        if df['Date'].dtype == 'object' or not pd.api.types.is_datetime64_any_dtype(df['Date']):
            st.write("Method 1 failed, trying Method 2...")
            df['Date'] = pd.to_datetime(df[date_column], errors='coerce')
            st.write(f"Method 2 type: {df['Date'].dtype}")
        
        # Try common date formats explicitly
        if df['Date'].dtype == 'object' or not pd.api.types.is_datetime64_any_dtype(df['Date']):
            st.write("Method 2 failed, trying specific formats...")
            
            formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y']
            
            for fmt in formats:
                try:
                    df['Date'] = pd.to_datetime(df[date_column], format=fmt, errors='coerce')
                    if pd.api.types.is_datetime64_any_dtype(df['Date']):
                        st.write(f"✅ Format {fmt} worked! Type: {df['Date'].dtype}")
                        break
                except:
                    continue
        
        # Count valid conversions
        valid_count = df['Date'].notna().sum()
        
        st.write(f"Conversion result: {valid_count}/{len(df)} dates converted")
        st.write(f"Final Date column type: {df['Date'].dtype}")
        
        if valid_count < len(df) * 0.5:
            st.error(f"❌ Only {valid_count} out of {len(df)} dates could be converted")
            st.write("Sample of original dates:", df[date_column].head(10).tolist())
            return None
        
        # CRITICAL: Check if it's actually datetime64
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            st.error(f"❌ Date column is still not datetime64! Type: {df['Date'].dtype}")
            
            # Last resort: manual conversion
            st.write("Attempting last resort conversion...")
            df['Date'] = df['Date'].astype('datetime64[ns]')
            st.write(f"After forced conversion: {df['Date'].dtype}")
            
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                st.error("❌ All conversion methods failed!")
                return None
        
        # Remove rows with NaT (Not a Time)
        initial_len = len(df)
        df = df.dropna(subset=['Date'])
        if len(df) < initial_len:
            st.warning(f"⚠️ Removed {initial_len - len(df)} rows with invalid dates")
        
        # Drop the original date column if different
        if date_column != 'Date' and date_column in df.columns:
            df = df.drop(columns=[date_column])
        
        # VERIFY the Date column is datetime64
        st.write(f"🔍 Date column type after conversion: {df['Date'].dtype}")
        st.write(f"🔍 Is datetime64? {pd.api.types.is_datetime64_any_dtype(df['Date'])}")
        
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            st.error(f"❌ Date column is not datetime64! Type: {df['Date'].dtype}")
            return None
        
        st.success(f"✅ Date conversion successful! Type: {df['Date'].dtype}")
        
        # Handle outliers for numeric columns only
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_columns:
            if column in df.columns and df[column].notna().sum() > 0:
                try:
                    median = df[column].median()
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                    if outlier_mask.sum() > 0:
                        df.loc[outlier_mask, column] = median
                except:
                    pass  # Skip if there's an error
        
        # Feature engineering with explicit check
        st.write("🔧 Creating date features...")
        st.write(f"About to use .dt accessor on type: {df['Date'].dtype}")
        
        # Create features one by one with error handling
        try:
            df['Year'] = df['Date'].dt.year
            st.write("✅ Year created")
        except Exception as e:
            st.error(f"❌ Failed to create Year: {e}")
            return None
            
        try:
            df['Month'] = df['Date'].dt.month
            st.write("✅ Month created")
        except Exception as e:
            st.error(f"❌ Failed to create Month: {e}")
            return None
            
        try:
            df['Day'] = df['Date'].dt.day
            st.write("✅ Day created")
        except Exception as e:
            st.error(f"❌ Failed to create Day: {e}")
            return None
            
        try:
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            st.write("✅ DayOfWeek created")
        except Exception as e:
            st.error(f"❌ Failed to create DayOfWeek: {e}")
            return None
            
        try:
            df['ElapsedDays'] = (df['Date'] - df['Date'].min()).dt.days
            st.write("✅ ElapsedDays created")
        except Exception as e:
            st.error(f"❌ Failed to create ElapsedDays: {e}")
            return None
        
        st.success("✅ All date features created successfully!")
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.error("Full error traceback:")
        st.code(traceback.format_exc())
        return None

# Home Page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">📈 Amazon Stock Price Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 3rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Advanced Machine Learning Models for Stock Price Forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models", "3", "Linear Regression, ARIMA, LSTM")
    
    with col2:
        if st.session_state.data_loaded and st.session_state.stock_data is not None:
            st.metric("Data Points", f"{len(st.session_state.stock_data):,}", "Trading Days")
        else:
            st.metric("Data Points", "0", "Trading Days")
    
    with col3:
        st.metric("Features", "15+", "Engineered Features")
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Features
    
    - **📊 Data Overview**: Explore Amazon stock data with interactive visualizations
    - **📈 Visualizations**: Detailed charts and analysis of stock trends
    - **🤖 Model Training**: Train and compare multiple ML models
    - **🔮 Predictions**: Forecast future stock prices
    - **📉 Model Comparison**: Compare model performance metrics
    
    ### 🚀 Getting Started
    
    1. Upload your stock data CSV file below
    2. Explore the data visualizations
    3. Train models in the Model Training section
    4. Make predictions and compare model performance
    
    ### 📋 CSV File Requirements
    
    Your CSV file should contain the following columns:
    - **Date**: Date of the trading day (any common date format)
    - **Open**: Opening price
    - **High**: Highest price
    - **Low**: Lowest price
    - **Close**: Closing price
    - **Volume**: Trading volume
    - **Dividends** (optional): Dividend payments
    - **Stock Splits** (optional): Stock split information
    """)
    
    # File uploader
    st.markdown("### 📁 Load Data")
    
    # Option to upload or use default path
    data_source = st.radio(
        "Data Source",
        ["Upload CSV File", "Use Default Path"],
        help="Choose to upload a file or use the default data path"
    )
    
    # Date format helper
    with st.expander("📅 Date Format Help"):
        st.markdown("""
        If your dates aren't loading correctly, check your date format:
        
        **Common Date Formats:**
        - `YYYY-MM-DD` → 2024-01-15
        - `MM/DD/YYYY` → 01/15/2024  
        - `DD/MM/YYYY` → 15/01/2024
        - `YYYY/MM/DD` → 2024/01/15
        
        The app will automatically try to detect your date format.
        """)
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload Amazon stock data CSV file",
            type=['csv'],
            help="Upload a CSV file with columns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                try:
                    stock_data = load_data(uploaded_file)
                    if stock_data is not None:
                        st.session_state.stock_data = stock_data
                        st.session_state.data_loaded = True
                        st.success(f"✅ Data loaded successfully! {len(stock_data):,} rows loaded.")
                        
                        # Display basic info
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Data Shape:**", stock_data.shape)
                            st.write("**Date Range:**", 
                                   f"{stock_data['Date'].min().strftime('%Y-%m-%d')} to {stock_data['Date'].max().strftime('%Y-%m-%d')}")
                        with col2:
                            st.write("**Columns:**", list(stock_data.columns[:8]))  # Show first 8 columns
                            st.write("**Missing Values:**", stock_data.isnull().sum().sum())
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    else:
        # Try to load from default path
        default_path = r"C:\Users\Nullvoid\Downloads\AMZN_stock_data.csv"
        
        st.info(f"Default path: `{default_path}`")
        st.write("Make sure the file exists at this location or modify the path below:")
        
        custom_path = st.text_input("Custom file path (optional):", value=default_path)
        
        if st.button("Load Data from Path", type="primary"):
            with st.spinner("Loading data..."):
                try:
                    stock_data = load_data(custom_path)
                    if stock_data is not None:
                        st.session_state.stock_data = stock_data
                        st.session_state.data_loaded = True
                        st.success(f"✅ Data loaded successfully! {len(stock_data):,} rows loaded.")
                        
                        # Display basic info
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Data Shape:**", stock_data.shape)
                            st.write("**Date Range:**", 
                                   f"{stock_data['Date'].min().strftime('%Y-%m-%d')} to {stock_data['Date'].max().strftime('%Y-%m-%d')}")
                        with col2:
                            st.write("**Columns:**", list(stock_data.columns[:8]))
                            st.write("**Missing Values:**", stock_data.isnull().sum().sum())
                except FileNotFoundError:
                    st.error(f"❌ File not found at: `{custom_path}`")
                    st.info("Please check the file path and try again, or upload a file instead.")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

# Data Overview Page
elif page == "📊 Data Overview":
    st.markdown('<h1 class="sub-header">📊 Data Overview</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data in the Home page first.")
        st.info("Go to 🏠 Home page to upload your stock data CSV file.")
    else:
        stock_data = st.session_state.stock_data
        
        # Data summary
        st.subheader("Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(stock_data):,}")
        with col2:
            days = (stock_data['Date'].max() - stock_data['Date'].min()).days
            st.metric("Date Range", f"{days:,} days")
        with col3:
            st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
        with col4:
            st.metric("Average Price", f"${stock_data['Close'].mean():.2f}")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(stock_data.head(10), use_container_width=True)
        
        # Data statistics
        st.subheader("Statistical Summary")
        st.dataframe(stock_data.describe(), use_container_width=True)
        
        # Data info
        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types:**")
            st.write(stock_data.dtypes)
        with col2:
            st.write("**Memory Usage:**")
            st.write(f"{stock_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        st.subheader("Missing Values")
        missing_data = stock_data.isnull().sum()
        if missing_data.sum() > 0:
            fig = px.bar(
                x=missing_data[missing_data > 0].index,
                y=missing_data[missing_data > 0].values,
                labels={'x': 'Column', 'y': 'Missing Count'},
                title='Missing Values by Column'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values in the dataset!")

# Visualizations Page
elif page == "📈 Visualizations":
    st.markdown('<h1 class="sub-header">📈 Data Visualizations</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data in the Home page first.")
    else:
        stock_data = st.session_state.stock_data
        
        # Visualization options
        viz_option = st.selectbox(
            "Select Visualization",
            ["Price Over Time", "Volume Analysis", "Price Distribution", "Correlation Matrix", "Daily Returns", "Candlestick Chart"]
        )
        
        if viz_option == "Price Over Time":
            st.subheader("Stock Price Over Time")
            fig = px.line(stock_data, x='Date', y='Close', title='Amazon Stock Closing Price Over Time')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional price metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Highest Price", f"${stock_data['Close'].max():.2f}")
            with col2:
                st.metric("Lowest Price", f"${stock_data['Close'].min():.2f}")
            with col3:
                price_range = stock_data['Close'].max() - stock_data['Close'].min()
                st.metric("Price Range", f"${price_range:.2f}")
        
        elif viz_option == "Volume Analysis":
            st.subheader("Trading Volume Analysis")
            fig = px.bar(stock_data, x='Date', y='Volume', title='Trading Volume Over Time')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Volume", f"{stock_data['Volume'].mean():,.0f}")
            with col2:
                st.metric("Max Volume", f"{stock_data['Volume'].max():,.0f}")
        
        elif viz_option == "Price Distribution":
            st.subheader("Price Distribution")
            fig = px.histogram(stock_data, x='Close', nbins=50, title='Distribution of Closing Prices')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Correlation Matrix":
            st.subheader("Feature Correlation Matrix")
            numeric_cols = stock_data.select_dtypes(include=[np.number]).columns
            correlation_matrix = stock_data[numeric_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                labels=dict(color="Correlation"),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='RdBu',
                aspect="auto",
                zmin=-1,
                zmax=1
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Daily Returns":
            st.subheader("Daily Returns")
            stock_data_copy = stock_data.copy()
            stock_data_copy['Daily_Return'] = stock_data_copy['Close'].pct_change()
            fig = px.line(stock_data_copy, x='Date', y='Daily_Return', title='Daily Returns Over Time')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                avg_return = stock_data_copy['Daily_Return'].mean() * 100
                st.metric("Average Daily Return", f"{avg_return:.2f}%")
            with col2:
                volatility = stock_data_copy['Daily_Return'].std() * 100
                st.metric("Volatility (Std Dev)", f"{volatility:.2f}%")
        
        elif viz_option == "Candlestick Chart":
            st.subheader("Candlestick Chart (Last 100 Days)")
            recent_data = stock_data.tail(100)
            fig = go.Figure(data=[go.Candlestick(
                x=recent_data['Date'],
                open=recent_data['Open'],
                high=recent_data['High'],
                low=recent_data['Low'],
                close=recent_data['Close']
            )])
            fig.update_layout(
                title='Stock Price Candlestick Chart',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

# Model Training Page
elif page == "🤖 Model Training":
    st.markdown('<h1 class="sub-header">🤖 Model Training</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data in the Home page first.")
    else:
        stock_data = st.session_state.stock_data
        
        st.subheader("Select Model to Train")
        model_options = ["Linear Regression"]
        if ARIMA_AVAILABLE:
            model_options.append("ARIMA")
        if LSTM_AVAILABLE:
            model_options.append("LSTM")
        model_options.append("Train All Models")
        
        model_choice = st.selectbox("Choose a model", model_options)
        
        # Model-specific parameters
        if model_choice == "LSTM" and LSTM_AVAILABLE:
            col1, col2, col3 = st.columns(3)
            with col1:
                seq_length = st.slider("Sequence Length", 30, 120, 60)
            with col2:
                epochs = st.slider("Epochs", 10, 100, 50)
            with col3:
                batch_size = st.slider("Batch Size", 16, 64, 32)
        else:
            seq_length, epochs, batch_size = 60, 50, 32
        
        if st.button("Train Model", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if model_choice == "Linear Regression":
                    status_text.text("Training Linear Regression model...")
                    progress_bar.progress(25)
                    results = train_linear_regression_model(stock_data)
                    st.session_state.lr_model = results
                    st.session_state.models_trained['lr'] = True
                    progress_bar.progress(100)
                    status_text.text("✅ Linear Regression model trained successfully!")
                    
                    # Display metrics
                    st.subheader("Training Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Train R²", f"{results['metrics']['train']['r2']:.4f}")
                        st.metric("Test R²", f"{results['metrics']['test']['r2']:.4f}")
                    with col2:
                        st.metric("Train MAE", f"${results['metrics']['train']['mae']:.2f}")
                        st.metric("Test MAE", f"${results['metrics']['test']['mae']:.2f}")
                    with col3:
                        st.metric("Train RMSE", f"${results['metrics']['train']['rmse']:.2f}")
                        st.metric("Test RMSE", f"${results['metrics']['test']['rmse']:.2f}")
                    
                    # Plot predictions
                    st.subheader("Predictions vs Actual")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['y_test'].values,
                        y=results['test_predictions'],
                        mode='markers',
                        name='Predictions',
                        marker=dict(color='blue', opacity=0.6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[results['y_test'].min(), results['y_test'].max()],
                        y=[results['y_test'].min(), results['y_test'].max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title='Test Data: Actual vs Predicted',
                        xaxis_title='Actual Price ($)',
                        yaxis_title='Predicted Price ($)',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif model_choice == "ARIMA" and ARIMA_AVAILABLE:
                    status_text.text("Training ARIMA model... This may take a while.")
                    progress_bar.progress(25)
                    results = train_arima_model(stock_data)
                    st.session_state.arima_model = results
                    st.session_state.models_trained['arima'] = True
                    progress_bar.progress(100)
                    status_text.text("✅ ARIMA model trained successfully!")
                    
                    # Display metrics
                    st.subheader("Training Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test R²", f"{results['metrics']['test']['r2']:.4f}")
                    with col2:
                        st.metric("Test MAE", f"${results['metrics']['test']['mae']:.2f}")
                    with col3:
                        st.metric("Test RMSE", f"${results['metrics']['test']['rmse']:.2f}")
                    
                    # Plot predictions
                    st.subheader("ARIMA Predictions")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['test_data'].index,
                        y=results['test_data']['Close'].values,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=results['test_data'].index,
                        y=results['predictions'],
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red')
                    ))
                    fig.update_layout(
                        title='ARIMA: Actual vs Predicted Over Time',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif model_choice == "LSTM" and LSTM_AVAILABLE:
                    status_text.text("Training LSTM model... This may take several minutes.")
                    progress_bar.progress(10)
                    results = train_lstm_model(stock_data, seq_length=seq_length, epochs=epochs, batch_size=batch_size)
                    st.session_state.lstm_model = results
                    st.session_state.models_trained['lstm'] = True
                    progress_bar.progress(100)
                    status_text.text("✅ LSTM model trained successfully!")
                    
                    # Display metrics
                    st.subheader("Training Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Train R²", f"{results['metrics']['train']['r2']:.4f}")
                        st.metric("Test R²", f"{results['metrics']['test']['r2']:.4f}")
                    with col2:
                        st.metric("Train MAE", f"${results['metrics']['train']['mae']:.2f}")
                        st.metric("Test MAE", f"${results['metrics']['test']['mae']:.2f}")
                    with col3:
                        st.metric("Train RMSE", f"${results['metrics']['train']['rmse']:.2f}")
                        st.metric("Test RMSE", f"${results['metrics']['test']['rmse']:.2f}")
                    
                    # Plot training history
                    st.subheader("Training History")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=results['history']['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        y=results['history']['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='red')
                    ))
                    fig.update_layout(
                        title='Model Loss Over Epochs',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot predictions
                    st.subheader("Predictions vs Actual")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['y_test_actual'].flatten(),
                        y=results['test_predictions'].flatten(),
                        mode='markers',
                        name='Predictions',
                        marker=dict(color='purple', opacity=0.6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[results['y_test_actual'].min(), results['y_test_actual'].max()],
                        y=[results['y_test_actual'].min(), results['y_test_actual'].max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title='Test Data: Actual vs Predicted',
                        xaxis_title='Actual Price ($)',
                        yaxis_title='Predicted Price ($)',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif model_choice == "Train All Models":
                    status_text.text("Training all models... This may take a while.")
                    models_trained = []
                    
                    # Linear Regression
                    progress_bar.progress(10)
                    status_text.text("Training Linear Regression...")
                    try:
                        lr_results = train_linear_regression_model(stock_data)
                        st.session_state.lr_model = lr_results
                        st.session_state.models_trained['lr'] = True
                        models_trained.append("Linear Regression")
                    except Exception as e:
                        st.error(f"Error training Linear Regression: {str(e)}")
                    
                    # ARIMA
                    if ARIMA_AVAILABLE:
                        progress_bar.progress(40)
                        status_text.text("Training ARIMA...")
                        try:
                            arima_results = train_arima_model(stock_data)
                            st.session_state.arima_model = arima_results
                            st.session_state.models_trained['arima'] = True
                            models_trained.append("ARIMA")
                        except Exception as e:
                            st.error(f"Error training ARIMA: {str(e)}")
                    
                    # LSTM
                    if LSTM_AVAILABLE:
                        progress_bar.progress(70)
                        status_text.text("Training LSTM... This may take several minutes.")
                        try:
                            lstm_results = train_lstm_model(stock_data)
                            st.session_state.lstm_model = lstm_results
                            st.session_state.models_trained['lstm'] = True
                            models_trained.append("LSTM")
                        except Exception as e:
                            st.error(f"Error training LSTM: {str(e)}")
                    
                    progress_bar.progress(100)
                    status_text.text(f"✅ Training complete! Models trained: {', '.join(models_trained)}")
                    st.success(f"Successfully trained {len(models_trained)} model(s): {', '.join(models_trained)}")
            
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                progress_bar.empty()
                status_text.empty()
        
        # Display training status
        st.subheader("Model Training Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.models_trained['lr']:
                st.success("✅ Linear Regression: Trained")
            else:
                st.info("⏳ Linear Regression: Not trained")
        with col2:
            if ARIMA_AVAILABLE:
                if st.session_state.models_trained['arima']:
                    st.success("✅ ARIMA: Trained")
                else:
                    st.info("⏳ ARIMA: Not trained")
            else:
                st.warning("⚠️ ARIMA: Not available (install statsmodels)")
        with col3:
            if LSTM_AVAILABLE:
                if st.session_state.models_trained['lstm']:
                    st.success("✅ LSTM: Trained")
                else:
                    st.info("⏳ LSTM: Not trained")
            else:
                st.warning("⚠️ LSTM: Not available (install tensorflow)")

# Predictions Page
elif page == "🔮 Predictions":
    st.markdown('<h1 class="sub-header">🔮 Stock Price Predictions</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data in the Home page first.")
    else:
        # Check if any models are trained
        any_trained = any(st.session_state.models_trained.values())
        if not any_trained:
            st.warning("⚠️ Please train models first in the Model Training section.")
        else:
            st.subheader("Future Price Prediction")
            
            # Select model for prediction
            available_models = []
            if st.session_state.models_trained['lr']:
                available_models.append("Linear Regression")
            if st.session_state.models_trained.get('arima', False):
                available_models.append("ARIMA")
            if st.session_state.models_trained.get('lstm', False):
                available_models.append("LSTM")
            
            if available_models:
                selected_model = st.selectbox("Select Model for Prediction", available_models)
                
                # Number of days to predict
                days_ahead = st.slider("Days Ahead to Predict", 1, 30, 7)
                
                if st.button("Generate Predictions", type="primary"):
                    if selected_model == "Linear Regression" and st.session_state.lr_model:
                        st.info("Linear Regression predictions require feature values. Use the model comparison page to see test predictions.")
                    
                    elif selected_model == "ARIMA" and st.session_state.arima_model:
                        try:
                            model = st.session_state.arima_model['model']
                            future_predictions = model.forecast(steps=days_ahead)
                            
                            st.subheader(f"ARIMA Predictions for Next {days_ahead} Days")
                            
                            # Create prediction dataframe
                            last_date = st.session_state.stock_data['Date'].max()
                            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
                            predictions_df = pd.DataFrame({
                                'Date': future_dates,
                                'Predicted_Price': future_predictions
                            })
                            
                            st.dataframe(predictions_df, use_container_width=True)
                            
                            # Plot predictions
                            fig = go.Figure()
                            # Historical data
                            recent_data = st.session_state.stock_data.tail(100)
                            fig.add_trace(go.Scatter(
                                x=recent_data['Date'],
                                y=recent_data['Close'],
                                mode='lines',
                                name='Historical Prices',
                                line=dict(color='blue')
                            ))
                            # Predictions
                            fig.add_trace(go.Scatter(
                                x=predictions_df['Date'],
                                y=predictions_df['Predicted_Price'],
                                mode='lines+markers',
                                name='Predicted Prices',
                                line=dict(color='red', dash='dash')
                            ))
                            fig.update_layout(
                                title=f'Stock Price Predictions (Next {days_ahead} Days)',
                                xaxis_title='Date',
                                yaxis_title='Price ($)',
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating predictions: {str(e)}")
                    
                    elif selected_model == "LSTM" and st.session_state.lstm_model:
                        st.info("LSTM future predictions require sequence data. Use the model comparison page to see test predictions.")
            
            # Show test predictions if available
            st.subheader("Test Set Predictions")
            if st.session_state.models_trained['lr'] and st.session_state.lr_model:
                with st.expander("📊 Linear Regression Test Predictions"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test R²", f"{st.session_state.lr_model['metrics']['test']['r2']:.4f}")
                    with col2:
                        st.metric("Test MAE", f"${st.session_state.lr_model['metrics']['test']['mae']:.2f}")
                    with col3:
                        st.metric("Test RMSE", f"${st.session_state.lr_model['metrics']['test']['rmse']:.2f}")
            
            if st.session_state.models_trained.get('arima', False) and st.session_state.arima_model:
                with st.expander("📊 ARIMA Test Predictions"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test R²", f"{st.session_state.arima_model['metrics']['test']['r2']:.4f}")
                    with col2:
                        st.metric("Test MAE", f"${st.session_state.arima_model['metrics']['test']['mae']:.2f}")
                    with col3:
                        st.metric("Test RMSE", f"${st.session_state.arima_model['metrics']['test']['rmse']:.2f}")
            
            if st.session_state.models_trained.get('lstm', False) and st.session_state.lstm_model:
                with st.expander("📊 LSTM Test Predictions"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test R²", f"{st.session_state.lstm_model['metrics']['test']['r2']:.4f}")
                    with col2:
                        st.metric("Test MAE", f"${st.session_state.lstm_model['metrics']['test']['mae']:.2f}")
                    with col3:
                        st.metric("Test RMSE", f"${st.session_state.lstm_model['metrics']['test']['rmse']:.2f}")

# Model Comparison Page
elif page == "📉 Model Comparison":
    st.markdown('<h1 class="sub-header">📉 Model Comparison</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data in the Home page first.")
    else:
        # Check if any models are trained
        any_trained = any(st.session_state.models_trained.values())
        if not any_trained:
            st.warning("⚠️ Please train models first in the Model Training section.")
        else:
            st.subheader("Model Performance Comparison")
            
            # Collect metrics from all trained models
            comparison_data = []
            
            if st.session_state.models_trained['lr'] and st.session_state.lr_model:
                lr_metrics = st.session_state.lr_model['metrics']['test']
                comparison_data.append({
                    'Model': 'Linear Regression',
                    'R² Score': lr_metrics['r2'],
                    'MAE': lr_metrics['mae'],
                    'RMSE': lr_metrics['rmse']
                })
            
            if st.session_state.models_trained.get('arima', False) and st.session_state.arima_model:
                arima_metrics = st.session_state.arima_model['metrics']['test']
                comparison_data.append({
                    'Model': 'ARIMA',
                    'R² Score': arima_metrics['r2'],
                    'MAE': arima_metrics['mae'],
                    'RMSE': arima_metrics['rmse']
                })
            
            if st.session_state.models_trained.get('lstm', False) and st.session_state.lstm_model:
                lstm_metrics = st.session_state.lstm_model['metrics']['test']
                comparison_data.append({
                    'Model': 'LSTM',
                    'R² Score': lstm_metrics['r2'],
                    'MAE': lstm_metrics['mae'],
                    'RMSE': lstm_metrics['rmse']
                })
            
            if comparison_data:
                # Create comparison dataframe
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display comparison table
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("R² Score Comparison")
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y='R² Score',
                        title='R² Score by Model (Higher is Better)',
                        color='R² Score',
                        color_continuous_scale='Viridis',
                        text='R² Score'
                    )
                    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("MAE Comparison")
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y='MAE',
                        title='Mean Absolute Error by Model (Lower is Better)',
                        color='MAE',
                        color_continuous_scale='Reds_r',
                        text='MAE'
                    )
                    fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # RMSE Comparison
                st.subheader("RMSE Comparison")
                fig = px.bar(
                    comparison_df,
                    x='Model',
                    y='RMSE',
                    title='Root Mean Squared Error by Model (Lower is Better)',
                    color='RMSE',
                    color_continuous_scale='Blues_r',
                    text='RMSE'
                )
                fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Best model
                st.subheader("🏆 Best Model by Metric")
                best_r2 = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
                best_mae = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']
                best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best R² Score", best_r2, 
                             f"{comparison_df.loc[comparison_df['R² Score'].idxmax(), 'R² Score']:.4f}")
                with col2:
                    st.metric("Best MAE", best_mae,
                             f"${comparison_df.loc[comparison_df['MAE'].idxmin(), 'MAE']:.2f}")
                with col3:
                    st.metric("Best RMSE", best_rmse,
                             f"${comparison_df.loc[comparison_df['RMSE'].idxmin(), 'RMSE']:.2f}")
                
                # Detailed metrics
                st.subheader("Detailed Metrics")
                for idx, row in comparison_df.iterrows():
                    with st.expander(f"📊 {row['Model']} - Detailed Metrics"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{row['R² Score']:.4f}")
                        with col2:
                            st.metric("MAE", f"${row['MAE']:.2f}")
                        with col3:
                            st.metric("RMSE", f"${row['RMSE']:.2f}")
                        
                        # Show interpretation
                        st.markdown("**Interpretation:**")
                        if row['R² Score'] > 0.9:
                            st.success("Excellent model fit (R² > 0.9)")
                        elif row['R² Score'] > 0.7:
                            st.info("Good model fit (R² > 0.7)")
                        else:
                            st.warning("Model fit could be improved (R² < 0.7)")
            else:
                st.info("No models trained yet. Please train models in the Model Training section.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>📈 Amazon Stock Price Prediction Dashboard</p>
    <p>Built with Streamlit • Machine Learning • Data Science</p>
</div>
""", unsafe_allow_html=True)