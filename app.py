import os
import io
import json
import math
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import streamlit as st

st.set_page_config(page_title="BudgetWise", page_icon="💸", layout="wide")
st.markdown(
    """
    <style>
    .fade-in { animation: fadeIn 0.6s ease-in both; }
    @keyframes fadeIn { from {opacity:0; transform: translateY(4px);} to {opacity:1; transform:none;} }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(buffer)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(buffer)
    else:
        return pd.DataFrame()
    cols = {c.strip().lower(): c for c in df.columns}
    date_col = None
    for key in ["date", "transaction date", "posted date", "booking date", "time"]:
        for k, orig in cols.items():
            if key == k:
                date_col = orig
                break
        if date_col:
            break
    amount_col = None
    for key in ["amount", "amt", "debit", "money", "value"]:
        for k, orig in cols.items():
            if key == k:
                amount_col = orig
                break
        if amount_col:
            break
    category_col = None
    for key in ["category", "cat", "type", "label", "tag"]:
        for k, orig in cols.items():
            if key == k:
                category_col = orig
                break
        if category_col:
            break
    merchant_col = None
    for key in ["merchant", "description", "narration", "details", "payee", "counterparty"]:
        for k, orig in cols.items():
            if key == k:
                merchant_col = orig
                break
        if merchant_col:
            break
    if amount_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["amount_raw"] = pd.to_numeric(df[amount_col], errors="coerce")
    if date_col is not None:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        out["date"] = pd.NaT
    if category_col is not None:
        out["category"] = df[category_col].astype(str).str.strip().replace({"": np.nan})
    else:
        out["category"] = np.nan
    if merchant_col is not None:
        out["merchant"] = df[merchant_col].astype(str).str.strip().replace({"": np.nan})
    else:
        out["merchant"] = np.nan
    out = out.dropna(subset=["amount_raw"]).copy()
    if out.empty:
        return out
    neg_share = (out["amount_raw"] < 0).mean()
    if neg_share >= 0.5:
        spend = (-out["amount_raw"]).clip(lower=0)
    else:
        spend = out["amount_raw"].clip(lower=0)
    out["spend"] = spend
    out = out[out["spend"] > 0].copy()
    if out["date"].isna().all():
        today = pd.Timestamp.today().normalize()
        out["date"] = today
    out["year_month"] = out["date"].dt.to_period("M").astype(str)
    out["category"] = out["category"].fillna("Uncategorized")
    out["merchant"] = out["merchant"].fillna("Unknown")
    return out

@st.cache_data(show_spinner=False)
def compute_metrics(df: pd.DataFrame):
    if df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    monthly = df.groupby("year_month", as_index=False)["spend"].sum()
    try:
        monthly["_sort_key"] = pd.to_datetime(monthly["year_month"], format="%Y-%m")
        monthly = monthly.sort_values("_sort_key").drop(columns=["_sort_key"]) 
    except Exception:
        pass
    by_cat = df.groupby("category", as_index=False)["spend"].sum().sort_values("spend", ascending=False)
    by_merch_freq = df["merchant"].value_counts().reset_index()
    by_merch_freq.columns = ["merchant", "count"]
    summary = {
        "total_spend": float(df["spend"].sum()),
        "months": monthly["year_month"].tolist(),
        "month_spend": monthly["spend"].round(2).tolist(),
        "top_categories": by_cat.head(3).to_dict(orient="records"),
        "top_merchants": by_merch_freq.head(5).to_dict(orient="records"),
        "max_month": monthly.loc[monthly["spend"].idxmax()]["year_month"] if not monthly.empty else None,
    }
    return summary, by_cat, monthly

def build_advice_prompt(summary: dict) -> str:
    total = summary.get("total_spend", 0)
    months = summary.get("months", [])
    month_spend = summary.get("month_spend", [])
    max_month = summary.get("max_month")
    tops = summary.get("top_categories", [])
    tops_txt = ", ".join([f"{t['category']}: ${t['spend']:.2f}" for t in tops]) if tops else "None"
    ms = ", ".join([f"{m}:{s:.2f}" for m, s in zip(months, month_spend)])
    prompt = (
        "You are a personal finance coach. Analyze the user's expenses and give 3-5 short, actionable recommendations. "
        "Return bullet points starting with '- '. Avoid generic tips; tailor to the data.\n\n"
        f"Total spend: ${total:.2f}\n"
        f"Monthly spend: {ms}\n"
        f"Highest month: {max_month}\n"
        f"Top categories by spend: {tops_txt}\n"
        "Consider category reduction ideas, subscription audits, spending caps, and timing adjustments."
    )
    return prompt

def hf_inference_api(prompt: str, token: str, model: str = "google/flan-t5-base", max_new_tokens: int = 180, temperature: float = 0.2) -> str:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature}}
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) and isinstance(data[0], dict):
            out = data[0].get("generated_text") or data[0].get("summary_text")
            if isinstance(out, str) and out.strip():
                return out.strip()
        if isinstance(data, dict):
            out = data.get("generated_text") or data.get("summary_text")
            if isinstance(out, str) and out.strip():
                return out.strip()
    except Exception:
        pass
    return ""

def simple_rule_advice(summary: dict) -> str:
    lines = []
    total = summary.get("total_spend", 0)
    tops = summary.get("top_categories", [])
    months = summary.get("months", [])
    month_spend = summary.get("month_spend", [])
    if tops:
        top = tops[0]
        lines.append(f"- Focus on reducing {top['category']} by 10–15%. Set a monthly cap and track against it weekly.")
    if len(tops) >= 2:
        second = tops[1]
        lines.append(f"- Compare alternatives for {second['category']} (switch providers, downgrade plans, or bundle).")
    if months and month_spend:
        idx = int(np.argmax(month_spend))
        lines.append(f"- {months[idx]} was your priciest month. Shift large purchases to lower-spend months to smooth cash flow.")
    lines.append("- Identify 2 recurring charges to cancel or renegotiate this month.")
    if total > 0:
        est = max(5, int(total * 0.05 // 5 * 5))
        lines.append(f"- Automate a ${est} weekly transfer to savings; increase by 5% after two stable months.")
    return "\n".join(lines[:5])

def generate_advice(summary: dict, token_input: str) -> str:
    prompt = build_advice_prompt(summary)
    token = token_input or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    if token:
        out = hf_inference_api(prompt, token)
        if out:
            return out
    return simple_rule_advice(summary)

st.title("BudgetWise: AI-Powered Personal Finance Assistant")
st.caption("Upload your expense export to analyze spending, visualize trends, and get tailored budget advice.")

with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
    st.header("AI Settings")
    hf_token = st.text_input("Hugging Face API Token (optional)", value="", type="password", help="Only used client-side to call HF Inference API.")
    st.header("Display")
    currency = st.selectbox("Currency symbol", ["$", "₹", "€", "£"], index=0)

if up is None:
    st.info("Upload a CSV/XLSX with at least amount and date columns to begin.")
    st.stop()

try:
    raw_bytes = up.getvalue()
    df = load_data(raw_bytes, up.name)
except Exception as e:
    st.error("Could not read the file.")
    st.stop()

if df.empty:
    st.warning("No expense rows detected after parsing. Ensure your file contains an amount column and try again.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    min_d = pd.to_datetime(df["date"], errors="coerce").min()
    max_d = pd.to_datetime(df["date"], errors="coerce").max()
    if pd.isna(min_d) or pd.isna(max_d):
        today = pd.Timestamp.today().normalize()
        min_d = max_d = today
    date_range = st.date_input("Date range", value=(min_d.date(), max_d.date()))
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
summary, by_cat, monthly = compute_metrics(df)

left, right = st.columns([1, 1])
with left:
    st.subheader("Spending by Category")
    if not by_cat.empty:
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        top_n = by_cat.head(8).copy()
        others = by_cat.iloc[8:]["spend"].sum()
        if others > 0:
            top_n = pd.concat([top_n, pd.DataFrame({"category": ["Other"], "spend": [others]})], ignore_index=True)
        ax1.pie(top_n["spend"], labels=top_n["category"], autopct="%1.1f%%", startangle=90, counterclock=False)
        ax1.axis('equal')
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.pyplot(fig1, clear_figure=True, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No category information available.")

with right:
    st.subheader("Monthly Spending Trend")
    if not monthly.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(monthly["year_month"], monthly["spend"], color="#4F46E5")
        ax2.set_xlabel("Month")
        ax2.set_ylabel(f"Spend ({currency})")
        ax2.set_xticklabels(monthly["year_month"], rotation=45, ha="right")
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.pyplot(fig2, clear_figure=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No monthly data available.")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Spend", f"{currency}{summary.get('total_spend', 0):,.2f}")
with m2:
    count_tx = int(df.shape[0])
    st.metric("Transactions", f"{count_tx}")
with m3:
    distinct_cat = int(df["category"].nunique())
    st.metric("Categories", f"{distinct_cat}")
with m4:
    max_month = summary.get("max_month") or "-"
    st.metric("Peak Month", f"{max_month}")

st.subheader("Quick Insights")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top 3 Categories**")
    top_cats = pd.DataFrame(summary.get("top_categories", []))
    if not top_cats.empty:
        top_cats = top_cats.rename(columns={"category": "Category", "spend": "Spend"})
        top_cats["Spend"] = top_cats["Spend"].round(2)
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.table(top_cats)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No category data.")
with c2:
    st.markdown("**Top 5 Merchants (by frequency)**")
    top_merch = pd.DataFrame(summary.get("top_merchants", []))
    if not top_merch.empty:
        top_merch = top_merch.rename(columns={"merchant": "Merchant", "count": "Count"})
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.table(top_merch)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No merchant data.")

st.subheader("AI-Generated Budget Advice")
with st.spinner("Generating advice..."):
    advice = generate_advice(summary, hf_token)
if advice.strip():
    st.markdown(advice)
    st.balloons()
    st.download_button("Download advice", advice, file_name="budgetwise_advice.txt")
    st.download_button("Download summary (JSON)", data=json.dumps(summary, indent=2), file_name="budgetwise_summary.json")
else:
    st.write("No advice generated.")

with st.expander("Data Preview"):
    st.dataframe(df.head(200), use_container_width=True)
