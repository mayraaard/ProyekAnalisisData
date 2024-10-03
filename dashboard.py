import pandas as pd
import matplotlib as plt
import streamlit as st
import seaborn as sns
from babel.numbers import format_currency
sns.set(style='dark')

def create_daily_orders_df(df):

    daily_orders_df = df.resample(rule='D', on='order_approved_at').agg({
        "order_id": "nunique",
        "payment_value": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "payment_value": "revenue"
    }, inplace=True)

    return daily_orders_df


def create_total_order_items_df(df):
    # Menghitung jumlah produk berdasarkan kategori
    total_order_items_df = df.groupby("product_category_name_english")["product_id"].count().reset_index()
    total_order_items_df.rename(columns={
        "product_id": "products"
    }, inplace=True)
    total_order_items_df = total_order_items_df.sort_values(by='products', ascending=False)

    return total_order_items_df


def get_review_scores_df(df):
    scores_count = df['review_score'].value_counts().sort_values(ascending=False)
    most_frequent_score = scores_count.idxmax()

    return scores_count, most_frequent_score


all_df = pd.read_csv("all_data.csv")

# Dataset
datetime_cols = ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", "order_purchase_timestamp", "shipping_limit_date"]
all_df.sort_values(by="order_approved_at", inplace=True)
all_df.reset_index(inplace=True)

for col in datetime_cols:
    all_df[col] = pd.to_datetime(all_df[col])

min_date = all_df["order_approved_at"].min()
max_date = all_df["order_approved_at"].max()

# Sidebar
with st.sidebar:
    # Title
    st.title("ML-56 Mayra Rahma Dianti")

    # Date Range
    start_date, end_date = st.date_input(
        label="Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Main
main_df = all_df[(all_df["order_approved_at"] >= str(start_date)) &
                 (all_df["order_approved_at"] <= str(end_date))]

daily_orders_df = create_daily_orders_df(main_df)
total_order_items_df = create_total_order_items_df(main_df)
scores_count, most_frequent_score = get_review_scores_df(main_df)

# Title
st.header("E-Commerce Dataset Dashboard")

# Daily Orders
st.subheader("Daily Orders")

col1, col2 = st.columns(2)

with col1:
    total_order = daily_orders_df["order_count"].sum()
    st.markdown(f"Total Order: **{total_order}**")

with col2:
    total_revenue = format_currency(daily_orders_df["revenue"].sum(), "IDR", locale="id_ID")
    st.markdown(f"Total Revenue: **{total_revenue}**")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    daily_orders_df["order_approved_at"],
    daily_orders_df["order_count"],
    marker="o",
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", labelsize=15)
st.pyplot(fig)

# Order Items
# ----------- Top and Bottom Products by Sales -----------
st.header("Question 1: Product Sales Analysis")

# Aggregating product sales
total_order_items_df = all_df.groupby("product_category_name_english")["product_id"].count().reset_index()
total_order_items_df = total_order_items_df.rename(columns={"product_id": "products"})

# Top 10 Products
top_products = total_order_items_df.sort_values(by="products", ascending=False).head(10)

# Bottom 10 Products
bottom_products = total_order_items_df.sort_values(by="products", ascending=True).head(10)

# Plotting Top and Bottom Products
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Top Products Bar Chart
sns.barplot(data=top_products, x='product_category_name_english', y='products', ax=axes[0])
axes[0].set_title('Top 10 Products by Sales')
axes[0].set_xlabel('Product Category')
axes[0].set_ylabel('Number of Products Sold')
axes[0].tick_params(axis='x', rotation=45)

# Bottom Products Bar Chart
sns.barplot(data=bottom_products, x='product_category_name_english', y='products', ax=axes[1])
axes[1].set_title('Bottom 10 Products by Sales')
axes[1].set_xlabel('Product Category')
axes[1].set_ylabel('Number of Products Sold')
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
st.pyplot(fig)

# ----------- Customer Ratings Distribution -----------
st.header("Question 2: Customer Ratings Distribution")

# Count review scores
score_counts = all_df['review_score'].value_counts().sort_values(ascending=False)

# Bar colors for ratings
bar_colors = ["#EAEAEA", "#EAEAEA", "#EAEAEA", "#EAEAEA", "#4C8BF5"]

# Plotting Customer Ratings
fig = plt.figure(figsize=(10, 5))
sns.set(style="darkgrid")
sns.barplot(x=score_counts.index, y=score_counts.values, palette=bar_colors)

plt.title("Customer Ratings for Service Quality", fontsize=15)
plt.xlabel("Rating", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
st.pyplot(fig)
