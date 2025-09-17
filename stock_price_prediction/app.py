import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime
import plotly.graph_objs as go

# ===============================
# 🎨 Page Config
# ===============================
st.set_page_config(page_title=" Stock Prediction Pro", layout="wide", page_icon="📉")

# ===============================
# 📂 Load Model and Features
# ===============================
model = joblib.load("stock_model.pkl")
features = joblib.load("features.pkl")

# ===============================
# Custom CSS Styling with Grey Background and Highlighted Card
# ===============================
st.markdown("""
<style>
body {
    background-color: #4a4a4a;  /* Medium grey background */
    color: #fff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3 {
    font-weight: 700;
}
.card {
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    background: linear-gradient(145deg, #3b3b3b, #2f2f2f);
    transition: transform 0.3s ease;
    color: white;
}
.card:hover {
    transform: translateY(-6px);
}
/* Highlight for Current Price & Volume card with distinct blue hue for visibility */
.card.highlight {
    background: linear-gradient(135deg, #283e6b, #4361ee);
    color: white;
    font-weight: 800;
}
.bullish, .bearish, .neutral, .prediction {
    color: black;
    font-weight: 800;
    border-radius: 10px;
    padding: 15px;
    margin-top: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.bullish { background: linear-gradient(135deg, #00c853, #a5d6a7); }
.bearish { background: linear-gradient(135deg, #d50000, #ef9a9a); }
.neutral { background: linear-gradient(135deg, #1e88e5, #90caf9); }
.prediction { background: linear-gradient(135deg, #ff6f00, #ffcc80); }

/* Sidebar styling */
#root > div:nth-child(1) > div > div.workspace > div.sidebar-content {
    background: #383838;
    border-radius: 15px;
    padding: 10px 15px;
}

/* Scrollbar for sidebar */
div[data-testid="stSidebar"]::-webkit-scrollbar {
    width: 8px;
}
div[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background-color: #606060;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.title("🔍 Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Symbol:", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date:", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date:", value=pd.to_datetime(datetime.today().strftime('%Y-%m-%d')))
fetch_data = st.sidebar.button("Fetch & Predict")

# ===============================
# 🛠 Feature Engineering Function
# ===============================
def calculate_technical_indicators(df):
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    if isinstance(bb_std, pd.DataFrame):
        bb_std = bb_std.iloc[:, 0]
    else:
        bb_std = bb_std.squeeze()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()

    for i in [1, 2, 3, 5]:
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
    return df

# ===============================
# Main App Logic
# ===============================
if fetch_data:
    data = yf.download(ticker, start=start_date, end=end_date.strftime('%Y-%m-%d'))
    if not data.empty:
        df = calculate_technical_indicators(data).dropna()
        if not df.empty:
            latest = df.iloc[-1]

            closing_price = float(latest['Close'])
            open_price = float(latest['Open'])

            X_latest = latest[features].values.reshape(1, -1)
            predicted_price = model.predict(X_latest)[0]

            st.title(f"📈 {ticker} Stock Prediction Pro")
            col1, col2, col3 = st.columns([3, 3, 4])

            with col1:
                st.markdown(f"""
                <div class="card highlight">
                    <h2>Current Price & Volume</h2>
                    <p>📅 <b>{df.index[-1].strftime('%Y-%m-%d')}</b></p>
                    <p>💰 <b>${closing_price:.2f}</b></p>
                    <p>📊 <b>{int(latest['Volume']):,}</b></p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                change = closing_price - open_price
                if change > 0:
                    trend_class, trend_text = "bullish", f"🚀 Bullish (+{change:.2f})"
                elif change < 0:
                    trend_class, trend_text = "bearish", f"📉 Bearish ({change:.2f})"
                else:
                    trend_class, trend_text = "neutral", "⚖️ Neutral"
                st.markdown(f"""
                <div class="card {trend_class}">
                    <h2>Market Trend</h2>
                    <p>{trend_text}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="card prediction">
                    <h2>🔮 Predicted Next Day Price</h2>
                    <p><b>${predicted_price:.2f}</b></p>
                </div>
                """, unsafe_allow_html=True)

            # Plotly candlestick chart + moving averages
            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name="Candlesticks",
                increasing_line_color='green',
                decreasing_line_color='red',
            ))

            ma_colors = {'MA_5':'#FFD700', 'MA_10':'#00FFFF', 'MA_20':'#FF69B4', 'MA_50':'#8A2BE2'}
            for ma, color in ma_colors.items():
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ma],
                    mode='lines', name=ma,
                    line=dict(color=color, width=2)
                ))

            fig.update_layout(
                title=f"{ticker} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                plot_bgcolor="#060505",
                paper_bgcolor="#000000",
                font=dict(color="#fff")
            )

            # RSI subplot below main chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#FFA500')))
            fig_rsi.update_layout(
                yaxis=dict(range=[0, 100]),
                title="RSI (Relative Strength Index)",
                height=250,
                plot_bgcolor="#010101",
                paper_bgcolor="#000000",
                font=dict(color="#fff"),
                margin=dict(l=30, r=10, t=30, b=30)
            )

            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig_rsi, use_container_width=True)

        else:
            st.error("❌ Not enough data to generate sophisticated features.")
    else:
        st.error("❌ No data found for this ticker symbol within the selected date range.")
else:
    st.info("✨ Enter parameters in the sidebar and click 'Fetch & Predict' to begin.")
