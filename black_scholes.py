import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import norm

# ---------- Page Setup ----------
st.set_page_config(page_title="📈 Black-Scholes Model", layout="wide")

# ---------- Sidebar Heading ----------
with st.sidebar:
    st.markdown("""
        <h2 style='font-weight: bold; font-size: 24px;'>📈 Black-Scholes Model</h2>
        <p style='font-size: 18px; margin-top: -1px;'>
            Created by <a href='https://www.linkedin.com/in/aarav-tyagi-446730208/' target='_blank' style='color:#1E90FF;text-decoration:none;'>Aarav Tyagi</a>
        </p>
    """, unsafe_allow_html=True)

# ---------- Sidebar: Input Parameters ----------
with st.sidebar.expander("Black-Scholes Inputs", expanded=True):
    st.header("Option Pricing Inputs")
    S = st.number_input("Spot Price (S)", min_value=0.0, value=100.0, step=1.0)
    X = st.number_input("Strike Price (X)", min_value=0.0, value=100.0, step=1.0)
    T = st.number_input("Time to Maturity (in years)", min_value=0.01, value=1.0, step=0.1)
    R_percent = st.number_input("Risk-Free Interest Rate (in %)", min_value=0.0, value=5.0, step=0.1)
    R = R_percent / 100
    V_model = st.number_input("Volatility for Model (σ)", min_value=0.01, max_value=2.0, value=0.2)

with st.sidebar.expander("Heatmap Parameters", expanded=True):
    st.header("Heatmap Range Setup")
    spot_min = st.number_input("Min Spot Price", min_value=1.0, value=80.0, step=1.0)
    spot_max = st.number_input("Max Spot Price", min_value=1.0, value=120.0, step=1.0)
    vol_min = st.slider("Min Volatility (σ)", min_value=0.01, max_value=2.0, value=0.1)
    vol_max = st.slider("Max Volatility (σ)", min_value=0.01, max_value=2.0, value=0.3)

# ---------- Black-Scholes Formula ----------
def black_scholes(S, X, T, R, V):
    d1 = (math.log(S/X) + (R + 0.5 * V**2) * T) / (V * math.sqrt(T))
    d2 = d1 - V * math.sqrt(T)
    call = S * norm.cdf(d1) - X * math.exp(-R * T) * norm.cdf(d2)
    put = X * math.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put

# ---------- Main Heading ----------
st.markdown("<h1 style='font-weight: bold;'>Black-Scholes Pricing Model</h1>", unsafe_allow_html=True)

# ---------- Display Input Table ----------
input_data = {
    "Current Asset Price": [S],
    "Strike Price": [X],
    "Time to Maturity (Years)": [T],
    "Volatility (σ)": [V_model],
    "Risk-Free Interest Rate": [R]
}
df_input = pd.DataFrame(input_data)
st.table(df_input.style.format(precision=4))

# ---------- Display Calculated Call/Put Price ----------
call_price, put_price = black_scholes(S, X, T, R, V_model)

col_call, col_put = st.columns(2)

with col_call:
    st.markdown(f"""
        <div style="background-color:#4CAF50;padding:20px;border-radius:10px;text-align:center;color:white;">
            <h4>CALL Value</h4>
            <h2 style="font-weight:bold;">{call_price:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

with col_put:
    st.markdown(f"""
        <div style="background-color:#e57373;padding:20px;border-radius:10px;text-align:center;color:white;">
            <h4>PUT Value</h4>
            <h2 style="font-weight:bold;">{put_price:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

# ---------- Heatmap Data Preparation ----------
spot_prices = np.linspace(spot_min, spot_max, 10)
volatilities = np.linspace(vol_min, vol_max, 10)

call_matrix = np.zeros((len(volatilities), len(spot_prices)))
put_matrix = np.zeros((len(volatilities), len(spot_prices)))

for i, v in enumerate(volatilities):
    for j, s in enumerate(spot_prices):
        call, put = black_scholes(s, X, T, R, v)
        call_matrix[i, j] = call
        put_matrix[i, j] = put

call_df = pd.DataFrame(call_matrix, index=[f"{v:.2f}" for v in volatilities],
                       columns=[f"{s:.2f}" for s in spot_prices])
put_df = pd.DataFrame(put_matrix, index=[f"{v:.2f}" for v in volatilities],
                      columns=[f"{s:.2f}" for s in spot_prices])

# ---------- Plotting Heatmaps ----------
st.header("Options Price - Interactive Heatmaps")

col1, col2 = st.columns(2)

common_kwargs = {
    "annot": True,
    "fmt": ".2f",
    "annot_kws": {"size": 8},
    "linewidths": 0.5,
    "cbar_kws": {"shrink": 0.85},
    "xticklabels": True,
    "yticklabels": True
}

with col1:
    st.subheader("Call Price Heatmap")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(call_df, cmap="YlGn", ax=ax1, **common_kwargs)
    ax1.set_xlabel("Spot Price")
    ax1.set_ylabel("Volatility")
    ax1.set_title("CALL", fontsize=14, weight="bold")
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader("Put Price Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(put_df, cmap="YlOrRd", ax=ax2, **common_kwargs)
    ax2.set_xlabel("Spot Price")
    ax2.set_ylabel("Volatility")
    ax2.set_title("PUT", fontsize=14, weight="bold")
    plt.tight_layout()
    st.pyplot(fig2)