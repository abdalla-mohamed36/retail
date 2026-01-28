import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Retail Price Optimization", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b1c2d 0%, #ffffff 35%);
}
h1, h2, h3 {
    color: #0b4f9a;
    font-weight: 700;
}
.black-text {
    color: black !important;
}
[data-testid="stSidebar"] {
    background-color: #000814;
    color: white;
}
.stButton>button {
    background-color: #0b4f9a;
    color: white;
    font-weight: 600;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #083d7c;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("AI and ML.csv")
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna()

df = load_data()

page = st.sidebar.selectbox("Page", ["Home", "EDA", "Elasticity"])

if page == "Home":
    st.title("Retail Price Optimization")

    st.markdown('<h3 class="black-text">Sample of Dataset</h3>', unsafe_allow_html=True)
    st.dataframe(df.head())

    summary = df[["Original_Price", "Selling_Price", "Quantity", "Total_Amount"]].agg(
        ["min", "mean", "max"]
    ).T
    summary.columns = ["Minimum", "Average", "Maximum"]

    st.markdown('<h3 class="black-text">Key Statistics</h3>', unsafe_allow_html=True)
    st.dataframe(summary)

elif page == "EDA":
    st.title("Exploratory Data Analysis")

    monthly = df.groupby(df["Date"].dt.to_period("M")).agg(
        qty=("Quantity", "sum"),
        rev=("Total_Amount", "sum")
    ).reset_index()

    fig = plt.figure()
    plt.plot(monthly["Date"].astype(str), monthly["qty"])
    plt.xticks(rotation=45)
    plt.title("Monthly Quantity Sold")
    st.pyplot(fig)
    plt.close()

    fig = plt.figure()
    plt.plot(monthly["Date"].astype(str), monthly["rev"])
    plt.xticks(rotation=45)
    plt.title("Monthly Revenue")
    st.pyplot(fig)
    plt.close()

    top_products = df.groupby("Product_Name")["Quantity"].sum().sort_values(ascending=False).head(10)
    fig = plt.figure()
    plt.barh(top_products.index, top_products.values)
    plt.gca().invert_yaxis()
    plt.title("Top 10 Products by Quantity Sold")
    st.pyplot(fig)
    plt.close()

elif page == "Elasticity":
    st.title("Price Elasticity Simulator")

    product = st.selectbox(
        "Select Product",
        sorted(df["Product_Name"].unique())
    )

    orig_price = df[df["Product_Name"] == product]["Original_Price"].mean()

    st.markdown(
        f'<p class="black-text"><b>Original price for "{product}" is {orig_price:.2f}</b></p>',
        unsafe_allow_html=True
    )

    seg = df[df["Product_Name"] == product]
    seg = seg[(seg["Selling_Price"] > 0) & (seg["Quantity"] > 0)]

    seg["log_price"] = np.log(seg["Selling_Price"])
    seg["log_qty"] = np.log(seg["Quantity"])

    X = seg[["log_price"]]
    y = seg["log_qty"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    elasticity = model.coef_[0]
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.markdown('<h3 class="black-text">Model Performance</h3>', unsafe_allow_html=True)
    st.markdown(f'<p class="black-text">Elasticity: <b>{elasticity:.3f}</b></p>', unsafe_allow_html=True)
    st.markdown(f'<p class="black-text">R² (test): <b>{r2:.3f}</b></p>', unsafe_allow_html=True)
    st.markdown(f'<p class="black-text">RMSE (test): <b>{rmse:.3f}</b></p>', unsafe_allow_html=True)

    base_price = seg["Selling_Price"].mean()
    base_qty = seg.groupby(seg["Date"].dt.to_period("M"))["Quantity"].sum().mean()

    new_price = st.number_input(
        "Test new price",
        value=float(round(base_price, 2)),
        step=0.1
    )

    if st.button("Simulate Impact"):
        if np.isclose(new_price, base_price):
            old_rev = base_price * base_qty
            new_rev = old_rev
            delta = 0.0
        else:
            pct_price_change = (new_price - base_price) / base_price
            pct_qty_change = elasticity * pct_price_change
            new_qty = base_qty * (1 + pct_qty_change)

            old_rev = base_price * base_qty
            new_rev = new_price * new_qty
            delta = new_rev - old_rev

        st.markdown('<h3 class="black-text">Revenue Impact</h3>', unsafe_allow_html=True)
        st.markdown(f'<p class="black-text">Original revenue: <b>{old_rev:,.2f}</b></p>', unsafe_allow_html=True)
        st.markdown(f'<p class="black-text">New revenue: <b>{new_rev:,.2f}</b></p>', unsafe_allow_html=True)

        if delta > 0:
            st.success(f"Revenue increases by {delta:,.2f}")
        elif delta < 0:
            st.error(f"Revenue decreases by {delta:,.2f}")
        else:
            st.info("No price change → no revenue change")

    if st.button("Reset"):
        st.experimental_rerun()
