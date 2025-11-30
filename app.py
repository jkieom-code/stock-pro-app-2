import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- Configuration ---
st.set_page_config(
    page_title="StockPro: Advanced AI Analytics",
    page_icon="📈",
    layout="wide"
)

# --- CSS Styling (Pro Look) ---
st.markdown("""
<style>
    .metric-container {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .stMetric {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_data(ticker, start, end):
    """Loads historical stock data from yfinance."""
    data = yf.download(ticker, start=start, end=end)
    
    # FIX: Flatten MultiIndex columns if they exist (prevents 'Series.format' error)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    data.reset_index(inplace=True)
    return data

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- ADVANCED FEATURE ENGINEERING FUNCTION ---
def add_technical_features(data):
    """Adds multiple technical indicators used for charting and AI prediction features."""
    
    # 1. Moving Averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # 2. Relative Strength Index (RSI)
    data['RSI'] = calculate_rsi(data)
    
    # 3. Moving Average Convergence Divergence (MACD)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Bollinger Bands (BB)
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()

    # 5. Date Feature for the Model (Ordinal)
    data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)

    # CRITICAL: Drop rows with NaN values (due to rolling window calculations)
    # The training model can only use complete data points.
    data.dropna(inplace=True)
    
    return data

def train_model(data):
    """Trains a Random Forest Regressor using technical indicators as features."""
    
    # Define features (X) and target (y)
    feature_cols = ['Date_Ordinal', 'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MA50']
    X = data[feature_cols]
    y = data['Close']
    
    # Train Random Forest (robust and good performance without deep learning complexity)
    model = RandomForestRegressor(
        n_estimators=150,           # Increased trees for better accuracy
        max_depth=10,               # Limit depth to prevent overfitting
        random_state=42,
        min_samples_leaf=5          # Ensure enough data per leaf
    )
    model.fit(X, y)
    
    return model, feature_cols

# --- Sidebar Controls ---
st.sidebar.header("🔍 Stock Configuration")
ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
years = st.sidebar.slider("Historical Data (Years)", 2, 5, 3) # Increased min years for better MA200/MA50 calc
forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)

# Dates
start_date = date.today() - timedelta(days=years*365)
end_date = date.today()

# --- Main Logic ---
st.title(f"📈 StockPro: {ticker} Advanced Analysis")

if ticker:
    try:
        # Load Data
        raw_data = load_data(ticker, start_date, end_date)
        
        if raw_data.empty or len(raw_data) < 200:
            st.warning("Data not found or not enough history (need > 200 days) for advanced analysis. Please check ticker or increase historical years.")
            st.stop()
            
        # Apply Advanced Feature Engineering
        data = add_technical_features(raw_data.copy())
            
        # --- ROBUST DATA EXTRACTION ---
        # Get latest metrics from the *engineered* data
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        high_24h = float(data['High'].iloc[-1])
        low_24h = float(data['Low'].iloc[-1])
        volume = int(data['Volume'].iloc[-1])

        delta = current_price - prev_price
        delta_percent = (delta / prev_price) * 100
        
        # --- Dashboard Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${current_price:.2f}", f"{delta:.2f} ({delta_percent:.2f}%)")
        col2.metric("High (24h)", f"${high_24h:.2f}")
        col3.metric("Low (24h)", f"${low_24h:.2f}")
        col4.metric("Volume", f"{volume:,}")

        # --- Tabs for Content ---
        tab1, tab2, tab3 = st.tabs(["📊 Technical Charts", "🔮 Advanced AI Prediction", "💾 Raw Data (Features Included)"])

        with tab1:
            st.subheader("Interactive Candlestick with Bollinger Bands")
            
            # Candlestick Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data['Date'],
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'],
                name='Market Data'
            ))
            
            # Bollinger Bands
            col_ma1, col_bb = st.columns(2)
            if col_bb.checkbox("Show Bollinger Bands (20-Day)", value=True):
                fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Upper'], line=dict(color='gray', dash='dash'), name='Upper Band'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Middle'], line=dict(color='yellow'), name='Middle Band'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Lower'], line=dict(color='gray', dash='dash'), name='Lower Band'))

            # Moving Averages Checkboxes
            if col_ma1.checkbox("Show 50-Day MA", value=True):
                fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], mode='lines', name='50-Day MA', line=dict(color='orange')))
            
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # MACD and RSI Indicator Subplots
            
            # RSI Indicator
            st.subheader("Momentum Indicators: RSI and MACD")
            fig_rsi = go.Figure(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(height=300, template="plotly_dark", yaxis_title="RSI Score", title="Relative Strength Index (RSI)")
            st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD Indicator
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD Line', line=dict(color='blue', width=2)))
            fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], name='Signal Line', line=dict(color='red', width=1)))
            fig_macd.add_trace(go.Bar(x=data['Date'], y=data['MACD'] - data['Signal_Line'], name='Histogram', marker_color='green'))
            fig_macd.update_layout(height=300, template="plotly_dark", yaxis_title="MACD Value", title="MACD Crossover")
            st.plotly_chart(fig_macd, use_container_width=True)


        with tab2:
            st.subheader("AI Trend Forecast (Feature-Rich Model)")
            st.write(
                "This model uses an enhanced **Random Forest Regressor** which considers key technical indicators (RSI, MACD, MAs, Price, and Volume) in addition to time. "
                "This greatly improves the model's ability to capture momentum and trend changes for a more accurate prediction."
            )
            
            if st.button("Run Advanced Prediction Model"):
                with st.spinner("Training AI Model with 8+ Features..."):
                    model, feature_cols = train_model(data)
                    
                    # --- Create Future Input Data ---
                    last_date = data['Date'].iloc[-1]
                    
                    # Create an array of future dates
                    future_dates = [last_date + timedelta(days=x) for x in range(1, forecast_days + 1)]
                    
                    # Initialize the forecast DataFrame with the required features
                    last_row = data.iloc[-1]
                    future_df = pd.DataFrame(columns=feature_cols)
                    
                    # Populate the future data iteratively
                    for i, d in enumerate(future_dates):
                        # The prediction for day 'i' must be based on the predicted price of day 'i-1'
                        if i == 0:
                            # Use the last actual data point as the basis for the first day's prediction
                            last_known_close = last_row['Close']
                        else:
                            # Use the previously predicted closing price
                            last_known_close = future_df.iloc[i-1]['Close_Predicted']

                        # Simple approximation for future Open/High/Low/Volume: assume they are equal to the last known values
                        # (A full financial model would forecast these too, but this is a reasonable simplification)
                        new_row = {
                            'Date_Ordinal': d.toordinal(),
                            'Open': last_known_close, # Use last close as next open approximation
                            'High': last_known_close * 1.005, # Simple high/low buffer
                            'Low': last_known_close * 0.995,
                            'Volume': last_row['Volume'],
                            # For technical indicators, we use the last known values as a starting proxy
                            'RSI': last_row['RSI'], 
                            'MACD': last_row['MACD'],
                            'MA50': last_row['MA50'],
                        }
                        
                        future_df.loc[i] = new_row
                        
                        # Predict the close price for the new day
                        future_features = future_df.loc[[i]][feature_cols]
                        pred_close = model.predict(future_features)[0]
                        
                        future_df.loc[i, 'Close_Predicted'] = pred_close
                        
                    # --- Plot Forecast ---
                    fig_pred = go.Figure()
                    
                    # Historical
                    fig_pred.add_trace(go.Scatter(
                        x=data['Date'], y=data['Close'],
                        mode='lines', name='Historical Data',
                        line=dict(color='blue')
                    ))
                    
                    # Predicted
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates, y=future_df['Close_Predicted'],
                        mode='lines+markers', name='AI Forecast',
                        line=dict(color='red', dash='dot')
                    ))
                    
                    fig_pred.update_layout(
                        title=f"{ticker} Price Prediction (Next {forecast_days} Days)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Display Raw Forecast Data
                    st.dataframe(future_df[['Date_Ordinal', 'Close_Predicted']].rename(columns={'Date_Ordinal': 'Date', 'Close_Predicted': 'Predicted Price'}))

        with tab3:
            st.subheader("Raw Data Inspector (Includes Features)")
            st.dataframe(data.sort_values(by='Date', ascending=False))
            
            # CSV Download
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Dataset (with Features) as CSV",
                data=csv,
                file_name=f'{ticker}_stock_data_advanced.csv',
                mime='text/csv',
            )

    except Exception as e:
        # This catches general errors like bad ticker symbols or network issues
        st.error(f"Error fetching data: {e}")
else:
    # Initial message when no ticker is entered
    st.info("Enter a stock ticker in the sidebar to begin.")
