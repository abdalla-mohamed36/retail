# app.py - Retail Price Optimization (full app)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------
# Small CSS (blue + white)
# -------------------------
st.markdown(
    """
    <style>
    .stApp {
      background: linear-gradient(180deg, #e6f0ff 0%, #ffffff 40%, #ffffff 100%);
    }
    .block-container {
      padding-top: 1.2rem;
      padding-bottom: 1rem;
      color: #000000;
      font-family: "Segoe UI", Roboto, Arial, sans-serif;
    }
    h1,h2,h3,h4,h5,h6 { color: #0b4f9a !important; font-weight:700 !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0b4f9a 0%, #07366a 100%); color: #ffffff; }
    .stButton>button { background: #0b66d1; color: #fff; border-radius:8px; padding:0.45rem 0.9rem; font-weight:600; }
    .stButton>button:hover { background:#085ab0; transform: translateY(-1px); }
    .stDataFrame, .dataframe { background: rgba(255,255,255,0.95); border-radius:8px; padding:8px; border:1px solid rgba(11,102,209,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load dataset
# -------------------------
DATAFILE = "AI and ML.csv"  # make sure this exact file is in the folder
try:
    df = pd.read_csv(DATAFILE)
except FileNotFoundError:
    st.error(f"File '{DATAFILE}' not found in the app folder. Put the CSV in the same folder as app.py and re-run.")
    st.stop()

# -------------------------
# Basic cleaning & types
# -------------------------
# Standardize column names to remove stray spaces and ensure exact names
df.columns = [c.strip() for c in df.columns]

# Required columns check
required = ["Date", "Order_ID", "Customer_ID", "Product_Name", "Category",
            "Original_Price", "Selling_Price", "Quantity", "Total_Amount"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}. Rename columns if needed and re-run.")
    st.stop()

# Convert types
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Original_Price"] = pd.to_numeric(df["Original_Price"], errors="coerce")
df["Selling_Price"] = pd.to_numeric(df["Selling_Price"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Total_Amount"] = pd.to_numeric(df["Total_Amount"], errors="coerce")

# Derived columns
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["UnitPrice_from_total"] = (df["Total_Amount"] / df["Quantity"]).round(2)  # for quick check
# NOTE: we will use Selling_Price as the price variable per your instruction

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Home", "EDA", "Elasticity"])

# ------------------------
# HOME PAGE
# ------------------------
if page == "Home":
    st.title("Retail Price Optimization App")
    st.subheader("Dataset preview")
    st.dataframe(df.head())

    st.subheader("Quick numeric summary")
    st.write(df[["Original_Price", "Selling_Price", "Quantity", "Total_Amount"]].describe().T)

    st.write(
        "Use the **Elasticity** page to estimate how quantity responds to Selling_Price changes "
        "for a selected product (Product_Name + Category)."
    )

# ------------------------
# EDA PAGE (Upgraded)
# ------------------------
if page == "EDA":
    st.header("Exploratory Data Analysis")

    # ===== BASIC SUMMARIES =====
    st.subheader("Dataset Summary")
    st.write(f"Rows: {df.shape[0]:,}")
    st.write(f"Columns: {df.shape[1]}")
    st.write(f"Unique customers: {df['Customer_ID'].nunique():,}")
    st.write(f"Unique products: {df['Product_Name'].nunique():,}")
    st.write(f"Categories: {df['Category'].nunique():,}")

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # ===== 1. MONTHLY SALES TREND =====
    st.subheader("Monthly Sales Trend (Quantity & Revenue)")

    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('YearMonth').agg(
        qty=('Quantity', 'sum'),
        rev=('Total_Amount', 'sum')
    ).reset_index()

    # Quantity trend
    fig = plt.figure()
    plt.plot(monthly['YearMonth'], monthly['qty'])
    plt.xticks(rotation=45)
    plt.title("Monthly Quantity Sold")
    plt.ylabel("Units")
    st.pyplot(fig)

    # Revenue trend
    fig = plt.figure()
    plt.plot(monthly['YearMonth'], monthly['rev'])
    plt.xticks(rotation=45)
    plt.title("Monthly Revenue")
    plt.ylabel("Revenue")
    st.pyplot(fig)

    # ===== 2. TOP 10 PRODUCTS =====
    st.subheader("Top 10 Products by Quantity Sold")

    top_products = (
        df.groupby("Product_Name")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig = plt.figure()
    plt.barh(top_products.index, top_products.values)
    plt.title("Top 10 Products (Quantity)")
    plt.xlabel("Units Sold")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    # ===== 3. CATEGORY SHARE =====
    st.subheader("Category Sales Mix")

    cat_share = df.groupby("Category")["Quantity"].sum()

    fig = plt.figure()
    plt.pie(cat_share, labels=cat_share.index, autopct='%1.1f%%')
    plt.title("Category Sales Share")
    st.pyplot(fig)

    # ===== 4. PRICE vs QUANTITY SCATTER =====
    st.subheader("Price vs Quantity Scatter")

    fig = plt.figure()
    plt.scatter(df["Selling_Price"], df["Quantity"], alpha=0.2)
    plt.xlabel("Selling Price")
    plt.ylabel("Quantity")
    plt.title("Price vs Quantity")
    st.pyplot(fig)

    # ===== CORRELATION MAP (KEEPING YOUR ORIGINAL STYLE) =====
    st.subheader("Correlation Heatmap (Numeric Features)")

    numeric_cols = ["Original_Price", "Selling_Price", "Quantity", "Total_Amount", "Year", "Month"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    corr_df = df[numeric_cols].corr()

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(corr_df.values, interpolation='none', aspect='auto')
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_df.index)), corr_df.index)

    for (i, j), val in np.ndenumerate(corr_df.values):
        plt.text(j, i, f"{val:.2f}", ha='center', va='center')

    st.pyplot(fig)


# --- Updated Elasticity Page Code (with Monthly Quantity & Monthly Revenue Impact) ---

# NOTE: Insert this inside your Streamlit app where the Elasticity page logic goes.
# This version adds:
# 1. average monthly quantity
# 2. monthly revenue calculations
# 3. clear display of monthly impact

if page == "Elasticity":
    st.header("Price Elasticity (with Monthly Revenue Impact)")

    st.write(
        "Select a product. Regression is run on binned Selling_Price values to estimate elasticity."
    )

    # Dropdown — now single dropdown for Product
    product_list = sorted(df["Product_Name"].dropna().astype(str).unique())
    sel_product = st.selectbox("Product Name", [""] + product_list)

    if not sel_product:
        st.info("Select a product to compute elasticity.")

    else:
        seg_df = df[df["Product_Name"].astype(str) == sel_product].copy()
        st.write(f"Rows in selected product: {len(seg_df)}")

        # Detect categories for transparency
        cats = seg_df["Category"].dropna().astype(str).unique().tolist()
        st.write(f"Detected Category/Categories: {cats}")

        # Basic cleaning
        seg_df = seg_df[(seg_df["Selling_Price"] > 0) & (seg_df["Quantity"] > 0)]
        if len(seg_df) < 5:
            st.warning("Not enough valid rows to estimate elasticity.")
        else:
            # -------------------------------------------------
            # NEW: Compute Monthly Quantity
            # -------------------------------------------------
            seg_df_sorted = seg_df.sort_values("Date").copy()
            seg_df_sorted["YearMonth"] = seg_df_sorted["Date"].dt.to_period("M")

            monthly = seg_df_sorted.groupby("YearMonth").agg(
                monthly_qty=("Quantity", "sum"),
                monthly_rev=("Total_Amount", "sum"),
            ).reset_index()

            avg_monthly_qty = monthly["monthly_qty"].mean()
            st.subheader("Average Monthly Sales")
            st.write(f"Average monthly units sold: **{avg_monthly_qty:.0f} units/month**")

            # -------------------------------------------------
            # Price binning for regression
            # -------------------------------------------------
            seg_df["price_bin"] = seg_df["Selling_Price"].round(1)
            agg = seg_df.groupby("price_bin").agg(
                mean_price=("Selling_Price", "mean"),
                mean_qty=("Quantity", "mean"),
                n=("Quantity", "size"),
            ).reset_index().sort_values("mean_price")

            if agg.shape[0] < 3:
                st.warning("Not enough distinct price bins to estimate elasticity.")
                st.dataframe(agg)
            else:
                # Regression
                x = np.log(agg["mean_price"].values).reshape(-1, 1)
                y = np.log(agg["mean_qty"].values)

                model = LinearRegression()
                model.fit(x, y)

                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
                y_pred = model.predict(x)

                # Metrics
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))

                st.subheader("Regression Results")
                st.write(f"Elasticity: **{slope:.3f}**")
                st.write(f"R²: **{r2:.3f}**, RMSE: **{rmse:.3f}**")
                st.write(f"Number of price bins: **{len(agg)}**")
                st.dataframe(agg)

                # Plot
                fig = plt.figure()
                plt.scatter(np.log(agg["mean_price"]), np.log(agg["mean_qty"]), alpha=0.7)
                xs = np.linspace(np.log(agg["mean_price"].min()), np.log(agg["mean_price"].max()), 100)
                plt.plot(xs, intercept + slope * xs, linewidth=2)
                plt.xlabel("log(mean price)")
                plt.ylabel("log(mean quantity)")
                plt.title(f"Log-Log Fit — {sel_product}")
                st.pyplot(fig)

                # -------------------------------------------------
                # PRICE CHANGE IMPACT — Monthly Level
                # -------------------------------------------------
                base_price = float(seg_df["Selling_Price"].mean())
                st.write(f"Average historical selling price: **{base_price:.2f}**")

                new_price = st.number_input(
                    "New proposed selling price",
                    value=round(base_price, 2), min_value=0.01, step=0.01
                )

                if st.button("Calculate Impact"):

                    pct_price_change = (new_price - base_price) / base_price
                    pct_qty_change = slope * pct_price_change

                    # Cap extreme predictions
                    pct_qty_change_capped = float(np.clip(pct_qty_change, -0.9, 1.0))

                    # Monthly prediction
                    expected_monthly_qty = avg_monthly_qty * (1 + pct_qty_change_capped)

                    original_monthly_revenue = avg_monthly_qty * base_price
                    new_monthly_revenue = expected_monthly_qty * new_price
                    revenue_change = new_monthly_revenue - original_monthly_revenue

                    # Emoji and message
                    if pct_qty_change_capped < -0.02:
                        emoji = "📉"
                        msg = "Expected decrease in sales"
                    elif pct_qty_change_capped > 0.02:
                        emoji = "📈"
                        msg = "Expected increase in sales"
                    else:
                        emoji = "😐"
                        msg = "No meaningful effect expected"

                    st.subheader("Monthly Impact Prediction")
                    st.write(f"{emoji} {msg}")

                    st.write(f"Price change: **{pct_price_change * 100:.2f}%**")
                    st.write(f"Predicted quantity change: **{pct_qty_change_capped * 100:.2f}%** (capped)")

                    st.write(f"Original monthly revenue: **{original_monthly_revenue:,.2f}**")
                    st.write(f"Expected new monthly revenue: **{new_monthly_revenue:,.2f}**")

                    if revenue_change >= 0:
                        st.success(f"Monthly revenue impact: +{revenue_change:,.2f}")
                    else:
                        st.error(f"Monthly revenue impact: {revenue_change:,.2f}")

                    st.write("Note: Predictions are based on historical small price variations. Large jumps may be less reliable.")
