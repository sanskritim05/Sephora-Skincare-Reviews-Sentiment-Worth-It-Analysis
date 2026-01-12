from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# -------------------------
# Page config + light styling
# -------------------------
st.set_page_config(
    page_title="Sephora Skincare: What’s Actually Worth It?",
    layout="wide",
)



st.markdown(
    """
    <style>
      div[data-testid="metric-container"] {
          padding: 12px 14px;
          border-radius: 14px;
          background: rgba(0,0,0,0.03);
      }
      h1, h2, h3 { margin-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(".").resolve()
RANKINGS_PATH = PROJECT_ROOT / "outputs" / "product_rankings.csv"

# title + tntro
st.title("Sephora Skincare: What’s Actually Worth It?")
st.caption(
    "Ranks skincare products using customer review data. "
    "The **Worth It Score** combines sentiment from review text, Bayesian-adjusted ratings "
    "(to avoid overrating low-review products), review volume, and a price-based value component."
)

# data loading
@st.cache_data
def load_rankings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "worth_it_score", "n_reviews", "avg_rating",
        "adj_pos_rate", "adj_rating", "adj_rating_norm",
        "price_usd", "value_score"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["product_name", "brand_name", "worth_it_score"])
    df["product_name"] = df["product_name"].astype(str)
    df["brand_name"] = df["brand_name"].astype(str)
    return df


def compute_brand_summary(rankings: pd.DataFrame, min_products: int = 5) -> pd.DataFrame:
    r = rankings.copy()

    if "n_reviews" not in r.columns or r["n_reviews"].isna().all():
        r["n_reviews"] = 1

    def wavg(series, weights):
        w = np.asarray(weights, dtype=float)
        x = np.asarray(series, dtype=float)
        w = np.nan_to_num(w, nan=0.0)
        if w.sum() <= 0:
            return float(np.nanmean(x))
        return float(np.average(x, weights=w))

    brand = (
        r.groupby("brand_name", as_index=False)
        .apply(lambda g: pd.Series({
            "n_products": len(g),
            "total_reviews": float(g["n_reviews"].sum()),
            "brand_worth_it": wavg(g["worth_it_score"], g["n_reviews"]),
            "avg_product_rating": wavg(g["avg_rating"], g["n_reviews"]) if "avg_rating" in g else np.nan,
            "avg_adj_pos_rate": wavg(g["adj_pos_rate"], g["n_reviews"]) if "adj_pos_rate" in g else np.nan
        }))
        .reset_index(drop=True)
    )

    brand = brand[brand["n_products"] >= min_products].copy()
    return brand


# guard
if not RANKINGS_PATH.exists():
    st.error("Missing outputs/product_rankings.csv. Run the notebook to generate it, then refresh this page.")
    st.stop()

rankings = load_rankings(RANKINGS_PATH)

# tabs
tab_products, tab_brands = st.tabs(["Products", "Brands"])

# products tab
with tab_products:
    st.subheader("Products")
    st.caption("Filter and sort products, then select one to see how its score is constructed.")

    # filters
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns([1, 1, 1.2, 1.2])

    with f1:
        min_reviews = st.slider("Minimum reviews", 0, 500, 25, 5, key="prod_min_reviews")

    with f2:
        top_n = st.selectbox("Rows to show", [25, 50, 100, 150, 200], index=1, key="prod_rows_to_show")

    with f3:
        brands = ["(All brands)"] + sorted(rankings["brand_name"].unique().tolist())
        brand_choice = st.selectbox("Brand", brands, key="prod_brand_choice")

    with f4:
        search = st.text_input(
            "Search product name",
            placeholder="e.g. moisturizer, serum, sunscreen",
            key="prod_search"
        )

    # sorting
    st.markdown("### Sort")
    sort_choice = st.selectbox(
        "Sort products by",
        [
            "Worth It Score (high → low)",
            "Avg Rating (high → low)",
            "Most Reviews",
            "Cheapest Price",
        ],
        index=0,
        key="prod_sort_choice"
    )

    sort_map = {
        "Worth It Score (high → low)": ("worth_it_score", False),
        "Avg Rating (high → low)": ("avg_rating", False),
        "Most Reviews": ("n_reviews", False),
        "Cheapest Price": ("price_usd", True),
    }
    sort_col, asc = sort_map[sort_choice]

    # apply filters
    filtered = rankings.copy()

    if min_reviews > 0 and "n_reviews" in filtered.columns:
        filtered = filtered[filtered["n_reviews"] >= min_reviews]

    if brand_choice != "(All brands)":
        filtered = filtered[filtered["brand_name"] == brand_choice]

    if search.strip():
        filtered = filtered[filtered["product_name"].str.contains(search, case=False, na=False)]

    if filtered.empty:
        st.info("No products match your filters. Try adjusting them.")
        st.stop()

    # sort
    filtered = filtered.sort_values(sort_col, ascending=asc)

    # quick stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products shown", f"{len(filtered):,}")
    c2.metric("Top score", f"{filtered['worth_it_score'].max():.3f}")
    c3.metric("Median rating", f"{filtered['avg_rating'].median():.2f}")
    c4.metric("Median reviews", f"{int(filtered['n_reviews'].median())}")

    # chart
    st.markdown("### Highest Worth It Scores (top 10)")
    chart_df = filtered.head(10)[["product_name", "worth_it_score"]].set_index("product_name")
    st.bar_chart(chart_df)

    # table 
    st.markdown("### Product table")
    cols = [
        "product_name", "brand_name",
        "worth_it_score", "n_reviews",
        "avg_rating", "adj_pos_rate",
        "price_usd", "value_score"
    ]
    cols = [c for c in cols if c in filtered.columns]

    table = filtered[cols].head(top_n).copy()
    for c in ["worth_it_score", "avg_rating", "adj_pos_rate", "price_usd", "value_score"]:
        if c in table.columns:
            table[c] = table[c].round(3)

    display_table = table.rename(columns={
        "product_name": "Product",
        "brand_name": "Brand",
        "worth_it_score": "Worth It Score",
        "n_reviews": "Number of Reviews",
        "avg_rating": "Average Rating",
        "adj_pos_rate": "Positive Review Rate (Adjusted)",
        "price_usd": "Average Price (USD)",
        "value_score": "Value Score"
    })

    st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="filtered_products.csv",
        mime="text/csv",
        key="prod_download_table"
    )

    # product details
    st.markdown("---")
    st.subheader("Product details")

    max_rows = min(top_n, len(filtered))
    detail_choice = st.selectbox(
        "Select a product",
        filtered.head(max_rows)["product_name"].tolist(),
        key="prod_detail_choice"
    )

    row = filtered[filtered["product_name"] == detail_choice].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Worth It Score", f"{row['worth_it_score']:.3f}")
    d2.metric("Avg Rating", f"{row['avg_rating']:.2f}")
    d3.metric("Adjusted Positive Rate", f"{row['adj_pos_rate']:.2%}")
    d4.metric("Reviews", f"{int(row['n_reviews'])}")

    st.write("**Brand:**", row["brand_name"])
    st.write("**Average price (USD): $**", f"{row['price_usd']:.2f}")
    st.write("**Price-based value score (higher = cheaper relative to other products):**", f"{row['value_score']:.3f}")

# brands tab
with tab_brands:
    st.subheader("Brands")
    st.caption("Compare brands using review-weighted averages across their products.")

    c1, c2 = st.columns([1, 1.2])
    with c1:
        min_products = st.slider("Minimum products per brand", 1, 50, 5, 1, key="brand_min_products")

    with c2:
        brand_sort_choice = st.selectbox(
            "Sort brands by",
            [
                "Brand score (high → low)",
                "Average rating (high → low)",
                "Total reviews (high → low)",
                "Most products",
            ],
            index=0,
            key="brand_sort_choice"
        )

    brand_summary = compute_brand_summary(rankings, min_products=min_products)

    if brand_summary.empty:
        st.info("No brands meet the minimum product threshold. Try lowering it.")
        st.stop()

    sort_map = {
        "Brand score (high → low)": ("brand_worth_it", False),
        "Average rating (high → low)": ("avg_product_rating", False),
        "Total reviews (high → low)": ("total_reviews", False),
        "Most products": ("n_products", False),
    }
    sort_col, asc = sort_map[brand_sort_choice]
    brand_summary = brand_summary.sort_values(sort_col, ascending=asc).reset_index(drop=True)

    # chart
    st.markdown("### Highest brand scores (top 10)")
    chart_df = brand_summary.head(10)[["brand_name", "brand_worth_it"]].set_index("brand_name")
    st.bar_chart(chart_df)

    # table 
    st.markdown("### Brand table")
    show_n = st.selectbox(
        "Rows to show",
        [25, 50, 75,100],
        index=1,
        key="brand_rows_to_show"
    )

    btable = brand_summary.head(show_n).copy()

    for c in ["brand_worth_it", "avg_product_rating", "avg_adj_pos_rate"]:
        if c in btable.columns:
            btable[c] = btable[c].round(3)

    display_btable = btable.rename(columns={
        "brand_name": "Brand",
        "brand_worth_it": "Brand Score",
        "n_products": "Number of Products",
        "total_reviews": "Total Reviews",
        "avg_product_rating": "Average Rating",
        "avg_adj_pos_rate": "Positive Review Rate (Adjusted)"
    })

    st.dataframe(display_btable, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download brand table (CSV)",
        data=btable.to_csv(index=False).encode("utf-8"),
        file_name="brand_scores.csv",
        mime="text/csv",
        key="brand_download_table"
    )

    # brand details
    st.markdown("---")
    st.subheader("Brand details")
    st.caption("Select a brand to view summary metrics and the top products from that brand.")

    chosen_brand = st.selectbox(
        "Select a brand",
        brand_summary["brand_name"].tolist(),
        key="brand_detail_choice"
    )

    brow = brand_summary[brand_summary["brand_name"] == chosen_brand].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Brand score", f"{brow['brand_worth_it']:.3f}")
    m2.metric("Products counted", f"{int(brow['n_products'])}")
    m3.metric("Total reviews", f"{int(brow['total_reviews'])}")
    m4.metric("Average rating", f"{brow['avg_product_rating']:.2f}" if pd.notna(brow.get("avg_product_rating")) else "—")

    # top products for this brand
    st.markdown("### Top products from this brand")

    brand_products = rankings[rankings["brand_name"] == chosen_brand].copy()

    # max based on selected brand
    if "n_reviews" in brand_products.columns and not brand_products["n_reviews"].isna().all():
        max_reviews_in_brand = int(brand_products["n_reviews"].max())
    else:
        max_reviews_in_brand = 0

    max_reviews_in_brand = max(max_reviews_in_brand, 500)

    min_reviews_brand = st.slider(
        "Minimum reviews per product (brand list)",
        min_value=0,
        max_value=max_reviews_in_brand,
        value=min(100, max_reviews_in_brand),
        step=10,
        key="brand_min_reviews_per_product"
    )


    if "n_reviews" in brand_products.columns and min_reviews_brand > 0:
        brand_products = brand_products[brand_products["n_reviews"] >= min_reviews_brand]

    if brand_products.empty:
        st.info("No products from this brand meet the minimum review threshold.")
    else:
        brand_products = brand_products.sort_values("worth_it_score", ascending=False).head(25)

        pcols = [
            "product_name",
            "worth_it_score",
            "n_reviews",
            "avg_rating",
            "adj_pos_rate",
            "price_usd",
            "value_score",
        ]
        pcols = [c for c in pcols if c in brand_products.columns]

        ptable = brand_products[pcols].copy()
        for c in ["worth_it_score", "avg_rating", "adj_pos_rate", "price_usd", "value_score"]:
            if c in ptable.columns:
                ptable[c] = ptable[c].round(3)

        display_ptable = ptable.rename(columns={
            "product_name": "Product",
            "worth_it_score": "Worth It Score",
            "n_reviews": "Number of Reviews",
            "avg_rating": "Average Rating",
            "adj_pos_rate": "Positive Review Rate (Adjusted)",
            "price_usd": "Average Price (USD)",
            "value_score": "Value Score"
        })

        st.dataframe(display_ptable, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download top products for this brand (CSV)",
            data=ptable.to_csv(index=False).encode("utf-8"),
            file_name=f"{chosen_brand.lower().replace(' ', '_')}_top_products.csv",
            mime="text/csv",
            key="brand_download_top_products"
        )

