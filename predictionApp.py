# TSMC Interactive Dashboard with Inline Training and Live Prediction
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- Load raw price data ---
df_raw = pd.read_csv("tsmc2330-1.csv")
df_raw['Date'] = pd.to_datetime(df_raw['Date']) if 'Date' in df_raw.columns else pd.date_range(start='2020-01-01', periods=len(df_raw))
df_raw.set_index('Date', inplace=True)

# Compute technical indicators
df = df_raw.copy()
df['MA5'] = df['Close'].rolling(5).mean()
df['MA10'] = df['Close'].rolling(10).mean()
df['MA20'] = df['Close'].rolling(20).mean()
if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

# Plot price and MAs
st.title("TSMC 技術分析與預測工具")
price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
for ma in ['MA5','MA10','MA20']:
    price_fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode='lines', name=ma))
price_fig.update_layout(title='收盤價與移動平均線', xaxis_title='日期', yaxis_title='價格')
st.plotly_chart(price_fig, use_container_width=True)

# Plot MACD
macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal'))
macd_fig.update_layout(title='MACD 與訊號線', xaxis_title='日期')
st.plotly_chart(macd_fig, use_container_width=True)

# --- Prepare features and target ---
df_feat = df.copy()
lag_cols = []
# Create lagged features
def lag(col):
    lag_col = 'Lag1_'+col
    df_feat[lag_col] = df_feat[col].shift(1)
    lag_cols.append(lag_col)

for col in ['Close','RSI','MACD','MACD_Signal']:
    lag(col)
# Derived features
df_feat['Lag1_Return'] = df_feat['Lag1_Close'].pct_change().shift(1)
lag_cols.append('Lag1_Return')
df_feat['Volume_Change'] = df_feat['Volume'].pct_change().shift(1)
lag_cols.append('Volume_Change')
df_feat['MA5_diff'] = (df_feat['MA5'] - df_feat['MA10']) / df_feat['MA10']
lag_cols.append('MA5_diff')
df_feat['BB_Pressure'] = df_feat['Close'] / df_feat['BB_Upper']
lag_cols.append('BB_Pressure')
# Target
df_feat['Target'] = (df_feat['Close'] > df_feat['Close'].shift(1)).astype(int)
# Drop missing
df_clean = df_feat.dropna(subset=lag_cols+['Target'])
X = df_clean[lag_cols]
y = df_clean['Target']

# Train scaler and model
dp = X.replace([np.inf,-np.inf],np.nan).dropna()
y_clean = y.loc[dp.index]
scaler = StandardScaler().fit(dp)
X_scaled = scaler.transform(dp)
model = LogisticRegression(class_weight='balanced').fit(X_scaled, y_clean)

# Default values from last row
last = df_clean.iloc[-1]

# Input UI
st.subheader("輸入今日技術指標預測明日漲跌")
col1,col2 = st.columns(2)
with col1:
    close = st.number_input("今日收盤價", value=float(last['Lag1_Close']), format="%.2f")
    close_prev = st.number_input("昨日收盤價", value=float(df_clean.iloc[-2]['Close']), format="%.2f")
    rsi = st.number_input("RSI", value=float(last['Lag1_RSI']), format="%.2f")
    macd = st.number_input("MACD", value=float(last['Lag1_MACD']), format="%.2f")
    macd_signal = st.number_input("MACD_Signal", value=float(last['Lag1_MACD_Signal']), format="%.2f")
with col2:
    volume = st.number_input("今日成交量", value=float(df_clean.iloc[-1]['Volume']), format="%.0f")
    volume_prev = st.number_input("昨日成交量", value=float(df_clean.iloc[-2]['Volume']), format="%.0f")
    ma5 = st.number_input("MA5", value=float(df_clean.iloc[-1]['MA5']), format="%.2f")
    ma10 = st.number_input("MA10", value=float(df_clean.iloc[-1]['MA10']), format="%.2f")
    bb_upper = st.number_input("BB_Upper", value=float(df_clean.iloc[-1]['BB_Upper']), format="%.2f")

if st.button("預測明日漲跌"):
    # Build input features
    data = {
        'Lag1_Close': close,
        'Lag1_Return': (close - close_prev) / close_prev,
        'Lag1_RSI': rsi,
        'Lag1_MACD': macd,
        'Lag1_MACD_Signal': macd_signal,
        'Volume_Change': (volume - volume_prev) / volume_prev,
        'MA5_diff': (ma5 - ma10) / ma10,
        'BB_Pressure': close / bb_upper
    }
    input_df = pd.DataFrame([data])
    # Ensure columns match training feature order
    input_df = input_df[lag_cols]
    # Scale and predict
    X_in = scaler.transform(input_df)
    pred = model.predict(X_in)[0]
    prob = model.predict_proba(X_in)[0][1]
    st.write(f"## 預測: {'上漲' if pred==1 else '下跌'}  信心度: {prob:.2%}")
    # Show inputs
    st.markdown("### 原始輸入值")
    st.dataframe(input_df.T)
        # Show scaled
    with st.expander("標準化後數據"):
        st.dataframe(pd.DataFrame(X_in, columns=lag_cols).T)
