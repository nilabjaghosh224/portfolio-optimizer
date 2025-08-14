import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ----------------------------
# Portfolio Functions
# ----------------------------
def download_adj_close(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)

    # Handle MultiIndex columns if they appear
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Adj Close']
    else:
        data = data[['Adj Close']]

    # If only one ticker, make sure it's a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    # Ensure column names match tickers
    data.columns = tickers

    return data.dropna(how='all')

def compute_log_returns(adj_close_df):
    lr = np.log(adj_close_df / adj_close_df.shift(1))
    return lr.dropna(how='any')

def annualized_stats(log_returns):
    mu = log_returns.mean() * 252.0
    cov = log_returns.cov() * 252.0
    return mu.values, cov.values

def portfolio_return(weights, mu):
    return float(np.dot(weights, mu))

def portfolio_volatility(weights, cov):
    var = float(weights.T @ cov @ weights)
    return np.sqrt(max(var, 0.0))

def sharpe_ratio(weights, mu, cov, rf):
    vol = portfolio_volatility(weights, cov)
    if vol == 0:
        return -1e9
    return (portfolio_return(weights, mu) - rf) / vol

def max_sharpe_weights(mu, cov, rf, bounds):
    n = len(mu)
    w0 = np.array([1.0/n] * n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    def neg_sharpe(w): return -sharpe_ratio(w, mu, cov, rf)
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x

def min_variance_weights(cov, bounds):
    n = len(cov)
    w0 = np.array([1.0/n] * n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    def variance(w): return float(w.T @ cov @ w)
    res = minimize(variance, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x

def efficient_frontier(mu, cov, bounds, n_points=50):
    mu_min, mu_max = float(np.min(mu)), float(np.max(mu))
    targets = np.linspace(mu_min, mu_max, n_points)
    rets, vols = [], []
    for tr in targets:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, mu=mu: np.dot(w, mu) - tr},
        )
        n = len(mu)
        w0 = np.array([1.0/n] * n)
        res = minimize(lambda w: float(w.T @ cov @ w), w0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            rets.append(portfolio_return(res.x, mu))
            vols.append(portfolio_volatility(res.x, cov))
    return np.array(rets), np.array(vols)

def random_portfolios(mu, cov, n_portfolios=5000):
    rets, vols = [], []
    n = len(mu)
    for _ in range(n_portfolios):
        w = np.random.random(n)
        w /= np.sum(w)
        rets.append(portfolio_return(w, mu))
        vols.append(portfolio_volatility(w, cov))
    return np.array(rets), np.array(vols)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📈 Portfolio Optimization (Markowitz MPT)")
st.markdown("Optimize your portfolio using **Max Sharpe** and **Minimum Variance** strategies.")

# Sidebar inputs
with st.sidebar:
    st.header("Portfolio Settings")
    tickers_input = st.text_input("Enter tickers (comma separated)", "SPY,BND,GLD,QQQ,VTI")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]
    years = st.slider("Years of Historical Data", 1, 10, 5)
    rf_rate = st.number_input("Risk-Free Rate", 0.0, 0.1, 0.02, step=0.005)
    max_weight = st.slider("Max Weight per Asset", 0.1, 1.0, 0.4, step=0.05)
    n_random = st.slider("Number of Random Portfolios", 1000, 10000, 5000, step=500)
    run_button = st.button("🚀 Run Optimization")

if run_button:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(years * 365))
    bounds = tuple((0.0, max_weight) for _ in range(len(tickers)))

    with st.spinner("Downloading data and running optimizations..."):
        adj_close = download_adj_close(tickers, start_date, end_date)
        log_returns = compute_log_returns(adj_close)
        mu, cov = annualized_stats(log_returns)

        w_sharpe = max_sharpe_weights(mu, cov, rf_rate, bounds)
        ret_sharpe = portfolio_return(w_sharpe, mu)
        vol_sharpe = portfolio_volatility(w_sharpe, cov)
        sr_sharpe = sharpe_ratio(w_sharpe, mu, cov, rf_rate)

        w_mvp = min_variance_weights(cov, bounds)
        ret_mvp = portfolio_return(w_mvp, mu)
        vol_mvp = portfolio_volatility(w_mvp, cov)
        sr_mvp = sharpe_ratio(w_mvp, mu, cov, rf_rate)

        fr_rets, fr_vols = efficient_frontier(mu, cov, bounds, 50)
        rand_rets, rand_vols = random_portfolios(mu, cov, n_random)

    # Display results
    st.subheader("📊 Portfolio Performance")
    results_df = pd.DataFrame({
        "Portfolio": ["Max Sharpe", "MVP"],
        "Return": [ret_sharpe, ret_mvp],
        "Volatility": [vol_sharpe, vol_mvp],
        "Sharpe Ratio": [sr_sharpe, sr_mvp]
    })
    st.dataframe(results_df.style.format({"Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}"}))

    st.subheader("📈 Efficient Frontier")
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(rand_vols, rand_rets, c='grey', alpha=0.3, s=10, label="Random Portfolios")
    ax.plot(fr_vols, fr_rets, 'b--', linewidth=2, label="Efficient Frontier")
    ax.scatter(vol_sharpe, ret_sharpe, color='r', label="Max Sharpe", s=60)
    ax.scatter(vol_mvp, ret_mvp, color='g', label="MVP", s=60)
    ax.set_xlabel("Volatility (Annualized)")
    ax.set_ylabel("Expected Return (Annualized)")
    ax.legend()
    st.pyplot(fig)

    # Weight charts
    st.subheader("🔍 Portfolio Weights")
    fig1, ax1 = plt.subplots()
    ax1.bar(tickers, w_sharpe)
    ax1.set_title("Max Sharpe Weights")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(tickers, w_mvp)
    ax2.set_title("MVP Weights")
    st.pyplot(fig2)

    # Download CSV
    csv_buf = io.StringIO()
    out_df = pd.DataFrame({"Ticker": tickers, "Max Sharpe Weight": w_sharpe, "MVP Weight": w_mvp})
    out_df.to_csv(csv_buf, index=False)
    st.download_button("💾 Download Weights CSV", csv_buf.getvalue(), "portfolio_weights.csv", "text/csv")

else:
    st.info("Enter your portfolio settings on the left and click **Run Optimization**.")
