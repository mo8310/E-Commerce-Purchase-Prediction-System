import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── sklearn / imbalanced-learn ──────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss,
    make_scorer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

SEED   = 42
COLORS = px.colors.qualitative.Plotly

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛒 E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (same as before) ──────────────────────────────────────────────
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 46px; border-radius: 8px 8px 0 0;
        padding: 0 20px; font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px; padding: 18px 22px; margin: 6px 0;
        border-left: 4px solid #4fc3f7;
    }
    .metric-card h3 { color: #b3e5fc; font-size: 13px; margin: 0 0 6px 0; }
    .metric-card p  { color: #ffffff; font-size: 26px; font-weight: 700; margin: 0; }
    .section-header {
        background: linear-gradient(90deg, #1a237e, #283593);
        color: white; padding: 12px 20px; border-radius: 10px;
        margin: 20px 0 14px 0; font-size: 17px; font-weight: 700;
    }
    div[data-testid="stSidebar"] { background: #0d1117; }
    div[data-testid="stSidebar"] * { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING (FIXED CSV – no upload needed)
# ════════════════════════════════════════════════════════════════════════════
DATA_PATH = "./ecommerce_user_behavior_8000.csv"

@st.cache_data
def load_raw_data():
    """Load the provided CSV file."""
    df = pd.read_csv(DATA_PATH)
    # Ensure numeric columns are properly typed (some may be read as object)
    for col in df.columns:
        if col not in ["user_id", "gender", "device_type"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

raw_df = load_raw_data()

# ── Preprocessing function (same core logic, but returns intermediate steps) ─
@st.cache_data
def preprocess_full(df: pd.DataFrame):
    """Full preprocessing + feature engineering. Returns processed df and info."""
    df = df.drop(columns=["user_id"], errors="ignore").copy()
    df = df.dropna(subset=["purchase"])

    num_feats = ["age","time_on_site","pages_viewed","previous_purchases",
                 "cart_items","avg_session_time","bounce_rate"]
    bin_feats = ["discount_seen","ad_clicked","returning_user"]
    cat_feats = ["gender","device_type"]

    # Imputation
    for c in num_feats + bin_feats:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_feats:
        df[c] = df[c].fillna(df[c].mode()[0])

    # Encoding
    le_g = LabelEncoder(); le_d = LabelEncoder()
    df["gender_enc"]      = le_g.fit_transform(df["gender"])
    df["device_type_enc"] = le_d.fit_transform(df["device_type"])

    # Feature engineering
    df["engagement_score"]   = (df["pages_viewed"] * df["time_on_site"]) / (df["avg_session_time"] + 1e-5)
    df["cart_to_page_ratio"] = df["cart_items"] / (df["pages_viewed"] + 1e-5)
    df["loyal_buyer"]        = ((df["previous_purchases"] > 3) & (df["returning_user"] == 1)).astype(int)
    df["high_engagement"]    = (df["engagement_score"] > df["engagement_score"].median()).astype(int)

    # Outlier capping
    cap_cols = ["time_on_site","pages_viewed","cart_items",
                "avg_session_time","engagement_score","cart_to_page_ratio"]
    for c in cap_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        df[c] = df[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

    return df, num_feats, bin_feats, cat_feats

df_proc, num_feats, bin_feats, cat_feats = preprocess_full(raw_df)

# ── Time features addition (for EDA) ─────────────────────────────────────────
@st.cache_data
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    date_range = pd.date_range("2024-01-01", "2024-12-31", freq="h")
    ts = pd.to_datetime(rng.choice(date_range, size=len(df), replace=True)).sort_values()
    df = df.copy()
    df["timestamp"]   = ts.values
    df["date"]        = df["timestamp"].dt.date
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["month"]       = df["timestamp"].dt.month
    df["week"]        = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = df["timestamp"].dt.dayofweek >= 5
    df["quarter"]     = df["timestamp"].dt.quarter
    return df

df_ts = add_time_features(df_proc)

# ── Global sidebar filters (based on processed data) ───────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=64)
st.sidebar.title("E-Commerce Analytics")
st.sidebar.markdown("---")

st.sidebar.markdown("Global Filters")
gender_sel = st.sidebar.multiselect("Gender", df_proc["gender"].unique(),
                                    default=list(df_proc["gender"].unique()))
device_sel = st.sidebar.multiselect("Device Type", df_proc["device_type"].unique(),
                                    default=list(df_proc["device_type"].unique()))
age_range  = st.sidebar.slider("Age Range", int(df_proc["age"].min()),
                                int(df_proc["age"].max()),
                                (int(df_proc["age"].min()), int(df_proc["age"].max())))

mask = (
    df_proc["gender"].isin(gender_sel) &
    df_proc["device_type"].isin(device_sel) &
    df_proc["age"].between(*age_range)
)
df_f  = df_proc[mask].copy()
df_ts = df_ts[mask].copy()

# ── Hero header ──────────────────────────────────────────────────────────────
st.title("E-Commerce User Behavior Analytics")
st.caption("End-to-end pipeline · EDA · Time Series · Modeling (CV + SMOTE + Pipelines)")

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(f'<div class="metric-card"><h3>Total Users</h3><p>{len(df_f):,}</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><h3>Purchases</h3><p>{int(df_f["purchase"].sum()):,}</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><h3>Purchase Rate</h3><p>{df_f["purchase"].mean():.1%}</p></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><h3>Avg Cart Items</h3><p>{df_f["cart_items"].mean():.2f}</p></div>', unsafe_allow_html=True)
with c5: st.markdown(f'<div class="metric-card"><h3>Avg Time on Site</h3><p>{df_f["time_on_site"].mean():.1f} min</p></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_eda, tab_preprocess, tab_ts, tab_model = st.tabs([
    "EDA — Exploratory Analysis",
    "Preprocessing Pipeline",
    "Time Series Analysis",
    "Modeling & Evaluation",
])

# =============================================================================
# TAB 1: EDA (unchanged from original, but cleaned)
# =============================================================================
with tab_eda:
    st.markdown('<div class="section-header"> Exploratory Data Analysis</div>', unsafe_allow_html=True)

    all_numeric = num_feats + ["engagement_score","cart_to_page_ratio"]
    all_cat     = cat_feats + bin_feats

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("Select Columns to Analyse")
        if "chosen_num_key" not in st.session_state:
            st.session_state.chosen_num_key = all_numeric[:4]
        if "chosen_cat_key" not in st.session_state:
            st.session_state.chosen_cat_key = all_cat[:3]
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Select all numeric"):
                st.session_state.chosen_num_key = all_numeric
        with col_btn2:
            if st.button("Select all categorical"):
                st.session_state.chosen_cat_key = all_cat
        chosen_num = st.multiselect("Numeric columns", all_numeric, key="chosen_num_key")
        chosen_cat = st.multiselect("Categorical / Binary columns", all_cat, key="chosen_cat_key")
        chart_type = st.radio("Distribution chart", ["Histogram", "Box plot", "Violin"], horizontal=True)

    with col_right:
        st.markdown("Target Distribution")
        vc = df_f["purchase"].value_counts().reset_index()
        vc.columns = ["purchase","count"]
        vc["label"] = vc["purchase"].map({1:"Purchase", 0:"No Purchase"})
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Count","Proportion"],
                            specs=[[{"type":"bar"},{"type":"pie"}]])
        fig.add_trace(go.Bar(x=vc["label"], y=vc["count"],
                             marker_color=["#2ecc71","#e74c3c"],
                             text=vc["count"], textposition="outside"), row=1, col=1)
        fig.add_trace(go.Pie(labels=vc["label"], values=vc["count"],
                             marker_colors=["#2ecc71","#e74c3c"],
                             hole=0.4, textinfo="percent+label"), row=1, col=2)
        fig.update_layout(height=320, showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    if chosen_num:
        st.markdown("Numeric Feature Distributions")
        n_cols = min(4, len(chosen_num))
        n_rows = (len(chosen_num) + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=chosen_num, vertical_spacing=0.14)
        for i, col in enumerate(chosen_num):
            r, c = divmod(i, n_cols)
            data0 = df_f.loc[df_f["purchase"]==0, col].dropna()
            data1 = df_f.loc[df_f["purchase"]==1, col].dropna()
            if chart_type == "Histogram":
                fig.add_trace(go.Histogram(x=data0, name="No Purchase", marker_color="#e74c3c",
                                           opacity=0.6, nbinsx=30, showlegend=(i==0)), row=r+1, col=c+1)
                fig.add_trace(go.Histogram(x=data1, name="Purchase", marker_color="#2ecc71",
                                           opacity=0.6, nbinsx=30, showlegend=(i==0)), row=r+1, col=c+1)
            elif chart_type == "Box plot":
                for label, vals, color in [("No Purchase",data0,"#e74c3c"),("Purchase",data1,"#2ecc71")]:
                    fig.add_trace(go.Box(y=vals, name=label, marker_color=color,
                                         legendgroup=label, showlegend=(i==0)), row=r+1, col=c+1)
            else:
                for label, vals, color in [("No Purchase",data0,"#e74c3c"),("Purchase",data1,"#2ecc71")]:
                    fig.add_trace(go.Violin(y=vals, name=label, line_color=color, fillcolor=color,
                                            opacity=0.5, legendgroup=label, showlegend=(i==0)), row=r+1, col=c+1)
        fig.update_layout(height=280*n_rows, barmode="overlay",
                          legend=dict(orientation="h", y=1.02), margin=dict(t=60))
        st.plotly_chart(fig, use_container_width=True)

    if chosen_cat:
        st.markdown("Categorical / Binary Feature Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Value Counts**")
            n_cc = len(chosen_cat)
            fig = make_subplots(rows=1, cols=n_cc, subplot_titles=chosen_cat)
            for i, col in enumerate(chosen_cat):
                vc2 = df_f[col].astype(str).value_counts().reset_index()
                vc2.columns = ["val","cnt"]
                fig.add_trace(go.Bar(x=vc2["val"], y=vc2["cnt"],
                                     marker_color=COLORS[i % len(COLORS)],
                                     text=vc2["cnt"], textposition="outside",
                                     showlegend=False), row=1, col=i+1)
            fig.update_layout(height=350, margin=dict(t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Purchase Rate by Category**")
            fig = make_subplots(rows=1, cols=n_cc, subplot_titles=chosen_cat)
            for i, col in enumerate(chosen_cat):
                rate = df_f.groupby(col)["purchase"].mean().reset_index()
                rate.columns = ["val","rate"]
                fig.add_trace(go.Bar(x=rate["val"].astype(str), y=rate["rate"],
                                     marker_color=COLORS[i % len(COLORS)],
                                     text=(rate["rate"]*100).round(1).astype(str)+"%",
                                     textposition="outside", showlegend=False), row=1, col=i+1)
            fig.update_yaxes(range=[0,1.2])
            fig.update_layout(height=350, margin=dict(t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("Correlation Matrix")
    heat_cols = [c for c in (chosen_num or all_numeric) + ["purchase"] if c in df_f.columns]
    if len(heat_cols) >= 2:
        corr = df_f[heat_cols].corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0,
            text=corr.values, texttemplate="%{text}", textfont_size=9,
            colorbar=dict(title="r")))
        fig.update_layout(height=520, xaxis_tickangle=-35, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    if len(chosen_num) >= 2:
        st.markdown("Scatter Matrix")
        subset = df_f[chosen_num[:6] + ["purchase"]].dropna()
        subset["purchase"] = subset["purchase"].astype(int).astype(str)
        fig = px.scatter_matrix(subset, dimensions=chosen_num[:6], color="purchase",
                                color_discrete_map={"1":"#2ecc71","0":"#e74c3c"},
                                opacity=0.35, height=650)
        fig.update_traces(diagonal_visible=False, marker_size=3)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: PREPROCESSING PIPELINE (NEW)
# =============================================================================
with tab_preprocess:
    st.markdown('<div class="section-header"> Data Preprocessing Pipeline</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This section shows the **exact transformations** applied to the raw data before modeling and analysis.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Raw Data (first 10 rows)")
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.caption(f"**Shape:** {raw_df.shape}  |  **Missing values:** {raw_df.isnull().sum().sum()}")
    with col2:
        st.markdown("Preprocessed Data (first 10 rows)")
        st.dataframe(df_proc.head(10), use_container_width=True)
        st.caption(f"**Shape:** {df_proc.shape}  |  **Target column:** purchase (0/1)")
    
    st.markdown("---")
    st.markdown("Preprocessing Steps Applied")
    st.markdown("""
    1. **Drop `user_id`** (non‑predictive)  
    2. **Remove rows with missing `purchase`** (target)  
    3. **Imputation**  
       - Numeric features: median imputation  
       - Categorical features: mode imputation  
    4. **Label Encoding**  
       - `gender` → `gender_enc`  
       - `device_type` → `device_type_enc`  
    5. **Feature Engineering**  
       - `engagement_score` = pages_viewed × time_on_site / (avg_session_time + ε)  
       - `cart_to_page_ratio` = cart_items / (pages_viewed + ε)  
       - `loyal_buyer` = (previous_purchases > 3) & (returning_user == 1)  
       - `high_engagement` = engagement_score > median(engagement_score)  
    6. **Outlier Capping** (IQR method) on selected continuous features  
    7. **Final dataset** – ready for modeling and time series analysis.
    """)
    
    # Show missing values summary in raw data
    st.markdown("Missing Values (Raw Data)")
    miss = raw_df.isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        pct = (miss / len(raw_df) * 100).round(2)
        miss_df = pd.DataFrame({"Missing": miss, "Pct (%)": pct})
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(miss_df, use_container_width=True)
        with col_b:
            fig = px.bar(miss_df.reset_index(), x="index", y="Pct (%)",
                         text="Pct (%)", color="Pct (%)", color_continuous_scale="Oranges",
                         labels={"index":"Column"}, title="Missing % per Column")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=340, coloraxis_showscale=False, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values in the raw data (after initial loading).")

# =============================================================================
# TAB 3: TIME SERIES (identical to original, shortened for brevity)
# =============================================================================
with tab_ts:
    st.markdown('<div class="section-header">Time Series Analysis</div>', unsafe_allow_html=True)
    ts_col1, ts_col2 = st.columns([2, 1])
    with ts_col2:
        ma_window = st.slider("Moving-average window (days)", 3, 30, 7)
        ts_view   = st.selectbox("View", ["Daily", "Monthly", "Day of Week", "Hour of Day",
                                          "Weekend vs Weekday", "Cohort (New vs Returning)",
                                          "Session Heatmap"])
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    
    if ts_view == "Daily":
        daily = df_ts.groupby("date").agg(sessions=("purchase","count"), purchases=("purchase","sum")).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        daily[f"sessions_{ma_window}d"] = daily["sessions"].rolling(ma_window, center=True).mean()
        daily[f"purchases_{ma_window}d"] = daily["purchases"].rolling(ma_window, center=True).mean()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=[f"Daily Sessions (raw + {ma_window}d MA)",
                                            f"Daily Purchases (raw + {ma_window}d MA)"])
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["sessions"], mode="lines",
                                 line=dict(color="rgba(31,119,180,0.3)", width=1), name="Sessions (raw)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily["date"], y=daily[f"sessions_{ma_window}d"], mode="lines",
                                 line=dict(color="#1f77b4", width=2.5), name=f"Sessions {ma_window}d MA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["purchases"], mode="lines",
                                 line=dict(color="rgba(44,160,44,0.3)", width=1), name="Purchases (raw)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=daily["date"], y=daily[f"purchases_{ma_window}d"], mode="lines",
                                 line=dict(color="#2ca02c", width=2.5), name=f"Purchases {ma_window}d MA"), row=2, col=1)
        fig.update_layout(height=520, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    elif ts_view == "Monthly":
        monthly = df_ts.groupby("month").agg(sessions=("purchase","count"), purchases=("purchase","sum"),
                                             avg_time=("time_on_site","mean"), avg_cart=("cart_items","mean")).reset_index()
        monthly["purchase_rate"] = monthly["purchases"] / monthly["sessions"]
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["month_name"] = monthly["month"].apply(lambda x: month_names[x-1])
        fig = make_subplots(rows=2, cols=2, subplot_titles=["Monthly Sessions","Purchase Rate","Avg Time on Site","Avg Cart Items"])
        fig.add_trace(go.Bar(x=monthly["month_name"], y=monthly["sessions"], marker_color=COLORS[0], name="Sessions"), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["purchase_rate"], mode="lines+markers",
                                 line=dict(color=COLORS[1], width=2.5), marker_size=8, name="Purchase Rate"), row=1, col=2)
        fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["avg_time"], mode="lines+markers",
                                 line=dict(color=COLORS[2], width=2.5), marker_size=8, name="Avg Time"), row=2, col=1)
        fig.add_trace(go.Bar(x=monthly["month_name"], y=monthly["avg_cart"], marker_color=COLORS[3], name="Avg Cart"), row=2, col=2)
        fig.update_layout(height=560, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    elif ts_view == "Day of Week":
        dow = df_ts.groupby("day_of_week").agg(sessions=("purchase","count"), purchases=("purchase","sum")).reindex(dow_order).reset_index()
        dow["purchase_rate"] = dow["purchases"] / dow["sessions"]
        colors_dow = ["#e74c3c" if d in ["Saturday","Sunday"] else "#3498db" for d in dow["day_of_week"]]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Sessions by Day","Purchase Rate by Day"])
        fig.add_trace(go.Bar(x=dow["day_of_week"], y=dow["sessions"], marker_color=colors_dow, text=dow["sessions"], textposition="outside", name="Sessions"), row=1, col=1)
        fig.add_trace(go.Bar(x=dow["day_of_week"], y=dow["purchase_rate"], marker_color=colors_dow,
                             text=(dow["purchase_rate"]*100).round(1).astype(str)+"%", textposition="outside", name="Purchase Rate"), row=1, col=2)
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    elif ts_view == "Hour of Day":
        hourly = df_ts.groupby("hour").agg(sessions=("purchase","count"), purchases=("purchase","sum")).reset_index()
        hourly["purchase_rate"] = hourly["purchases"] / hourly["sessions"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Sessions by Hour","Purchase Rate by Hour"])
        fig.add_trace(go.Bar(x=hourly["hour"], y=hourly["sessions"], marker_color=COLORS[1], name="Sessions"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["purchase_rate"], mode="lines+markers",
                                 line=dict(color=COLORS[3], width=2.5), marker_size=7, name="Purchase Rate"), row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    elif ts_view == "Weekend vs Weekday":
        cols_wk = ["time_on_site","pages_viewed","cart_items","avg_session_time","bounce_rate"]
        wk = df_ts.groupby("is_weekend")[cols_wk].mean().reset_index()
        wk["is_weekend"] = wk["is_weekend"].map({True:"Weekend", False:"Weekday"})
        wk_melt = wk.melt(id_vars="is_weekend", var_name="Metric", value_name="Value")
        fig = px.bar(wk_melt, x="Metric", y="Value", color="is_weekend", barmode="group",
                     color_discrete_map={"Weekday":"#3498db","Weekend":"#e74c3c"}, text_auto=".2f")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=430, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
    elif ts_view == "Cohort (New vs Returning)":
        cohort = df_ts.groupby(["quarter","returning_user"]).agg(sessions=("purchase","count"), purchases=("purchase","sum"),
                                                                 avg_cart=("cart_items","mean")).reset_index()
        cohort["purchase_rate"] = cohort["purchases"] / cohort["sessions"]
        cohort["user_type"] = cohort["returning_user"].map({1.0:"Returning", 0.0:"New"})
        cohort["Q"] = "Q" + cohort["quarter"].astype(str)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Purchase Rate by Quarter","Avg Cart by Quarter"])
        for label, color in [("Returning","#2ecc71"),("New","#e74c3c")]:
            sub = cohort[cohort["user_type"]==label]
            fig.add_trace(go.Bar(x=sub["Q"], y=sub["purchase_rate"], name=label, marker_color=color, legendgroup=label,
                                 text=(sub["purchase_rate"]*100).round(1).astype(str)+"%", textposition="outside"), row=1, col=1)
            fig.add_trace(go.Bar(x=sub["Q"], y=sub["avg_cart"], name=label, marker_color=color, legendgroup=label, showlegend=False,
                                 text=sub["avg_cart"].round(2), textposition="outside"), row=1, col=2)
        fig.update_layout(height=450, barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    elif ts_view == "Session Heatmap":
        pivot = df_ts.groupby(["day_of_week","hour"]).size().reset_index(name="count")
        pivot_mat = pivot.pivot(index="day_of_week", columns="hour", values="count").reindex(dow_order)
        fig = go.Figure(go.Heatmap(z=pivot_mat.values, x=pivot_mat.columns, y=pivot_mat.index, colorscale="Viridis"))
        fig.update_layout(title="Session Heatmap: Hour of Day × Day of Week", xaxis_title="Hour", yaxis_title="Day", height=420)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 4: MODELING WITH CROSS-VALIDATION
# =============================================================================
with tab_model:
    st.markdown('<div class="section-header">Machine Learning Modeling & Evaluation</div>', unsafe_allow_html=True)

    ALL_FEATURES = (num_feats + bin_feats +
                    ["gender_enc","device_type_enc",
                     "engagement_score","cart_to_page_ratio","loyal_buyer","high_engagement"])

    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.markdown("Configuration")
        chosen_feats = st.multiselect("Feature columns", ALL_FEATURES, default=ALL_FEATURES)
        use_smote = st.checkbox("Apply SMOTE oversampling", value=True)
        cv_folds = st.slider("Number of CV folds (Stratified K-Fold)", 3, 10, 5)
        model_choices = st.multiselect(
            "Models to train",
            ["CatBoost","Random Forest","XGBoost","Gradient Boosting",
             "Decision Tree","KNN","Naive Bayes"],
            default=["CatBoost","Random Forest","XGBoost"]
        )
        train_btn = st.button("Train & Cross-Validate", type="primary", use_container_width=True)

    with m_col2:
        st.markdown("Feature Engineering Summary")
        fe_df = pd.DataFrame({
            "Feature": ["engagement_score","cart_to_page_ratio","loyal_buyer","high_engagement"],
            "Formula": [
                "pages × time / (avg_session + ε)",
                "cart_items / (pages_viewed + ε)",
                "prev_purch > 3 AND returning_user == 1",
                "engagement_score > median",
            ],
            "Type": ["Continuous","Continuous","Binary","Binary"]
        })
        st.dataframe(fe_df, use_container_width=True, hide_index=True)
        if chosen_feats:
            tc = df_f[chosen_feats + ["purchase"]].corr()["purchase"].drop("purchase").sort_values()
            bar_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in tc.values]
            fig = go.Figure(go.Bar(x=tc.values, y=tc.index, orientation="h", marker_color=bar_colors,
                                   text=tc.round(3).values, textposition="outside"))
            fig.update_layout(title="Feature ↔ Target Correlation", height=340, xaxis_title="Pearson r")
            st.plotly_chart(fig, use_container_width=True)

    if train_btn:
        if not chosen_feats:
            st.error("Please select at least one feature.")
            st.stop()
        if not model_choices:
            st.error("Please select at least one model.")
            st.stop()

        X = df_f[chosen_feats]
        y = df_f["purchase"].astype(int)

        # Preprocessor for numeric & categorical
        num_c = X.select_dtypes(include=["int64","float64"]).columns.tolist()
        cat_c = X.select_dtypes(include=["object"]).columns.tolist()
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("scl", StandardScaler())])
        transformers = [("num", num_pipe, num_c)]
        if cat_c:
            cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                 ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
            transformers.append(("cat", cat_pipe, cat_c))
        preprocessor = ColumnTransformer(transformers)

        model_map = {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6,
                                                    min_samples_leaf=10, class_weight="balanced", random_state=SEED),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                                     scale_pos_weight=(len(y)-y.sum())/max(y.sum(),1), eval_metric="logloss", random_state=SEED),
            "CatBoost": CatBoostClassifier(iterations=100, learning_rate=0.05, depth=3,
                                           auto_class_weights="Balanced", random_seed=SEED, verbose=False),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                                    class_weight="balanced", random_state=SEED),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                            max_depth=3, subsample=0.8, random_state=SEED),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
        }

        results = []  # will store cross-validation results
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        scorers = {"roc_auc": make_scorer(roc_auc_score, needs_proba=True),
                   "avg_precision": make_scorer(average_precision_score, needs_proba=True)}

        prog = st.progress(0, text="Running cross-validation...")
        for i, name in enumerate(model_choices):
            mdl = model_map[name]
            if use_smote:
                pipe = ImbPipeline([("prep", preprocessor),
                                    ("smote", SMOTE(random_state=SEED)),
                                    ("clf", mdl)])
            else:
                pipe = Pipeline([("prep", preprocessor), ("clf", mdl)])
            
            # Cross-validation scores
            cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            cv_ap = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
            
            results.append({
                "Model": name,
                "CV ROC-AUC mean": round(cv_auc.mean(), 4),
                "CV ROC-AUC std": round(cv_auc.std(), 4),
                "CV Avg Precision mean": round(cv_ap.mean(), 4),
                "CV Avg Precision std": round(cv_ap.std(), 4),
            })
            prog.progress((i+1)/len(model_choices), text=f"Evaluated {name}")
        prog.empty()

        # Create a comparison dataframe
        res_df = pd.DataFrame(results).sort_values("CV ROC-AUC mean", ascending=False)
        best_name = res_df.iloc[0]["Model"]
        
        st.success(f"Cross-validation complete! Best model based on mean ROC-AUC: **{best_name}** (CV ROC-AUC = {res_df.iloc[0]['CV ROC-AUC mean']:.4f} ± {res_df.iloc[0]['CV ROC-AUC std']:.4f})")
        
        # Display comparison table
        st.markdown("Model Cross-Validation Comparison")
        st.dataframe(res_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        
        # Bar plot of mean CV ROC-AUC
        fig = go.Figure()
        fig.add_trace(go.Bar(x=res_df["Model"], y=res_df["CV ROC-AUC mean"],
                             error_y=dict(type="data", array=res_df["CV ROC-AUC std"]),
                             marker_color=["#f39c12" if m == best_name else "#3498db" for m in res_df["Model"]],
                             text=res_df["CV ROC-AUC mean"], textposition="outside", name="ROC-AUC (CV)"))
        fig.update_layout(title="Cross-Validated ROC-AUC (mean ± std)", yaxis=dict(range=[0,1]), height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optional: train final best model on full data and show threshold tuning, confusion matrix, etc.
        st.markdown(f"Final Model Training (Best: {best_name}) on Full Data")
        st.markdown("The following analysis uses the best model trained on the entire dataset (with optional SMOTE) to show threshold tuning, feature importance, and confusion matrix.")
        
        # Re-train best model on all data (without CV split) for interpretability
        X_all = X
        y_all = y
        final_pipe = None
        if use_smote:
            final_pipe = ImbPipeline([("prep", preprocessor),
                                      ("smote", SMOTE(random_state=SEED)),
                                      ("clf", model_map[best_name])])
        else:
            final_pipe = Pipeline([("prep", preprocessor), ("clf", model_map[best_name])])
        final_pipe.fit(X_all, y_all)
        
        # For evaluation, we still need a holdout set to get unbiased metrics and curves.
        # Use a 70/30 train/test split (stratified) for final evaluation.
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, stratify=y_all, random_state=SEED)
        final_pipe.fit(X_train, y_train)
        y_proba = final_pipe.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)
        test_ap = average_precision_score(y_test, y_proba)
        
        st.markdown(f"**Test set performance (30% holdout):** ROC-AUC = {test_auc:.4f},  Avg Precision = {test_ap:.4f}")
        
        # Curves
        cur1, cur2 = st.columns(2)
        with cur1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig = go.Figure()
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={test_auc:.3f})"))
            fig.update_layout(title="ROC Curve (Holdout Test)", xaxis_title="FPR", yaxis_title="TPR", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with cur2:
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR (AP={test_ap:.3f})"))
            fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Threshold tuning
        prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, y_proba)
        f1_arr = 2*(prec_arr[:-1]*rec_arr[:-1])/(prec_arr[:-1]+rec_arr[:-1]+1e-8)
        opt_idx = int(np.argmax(f1_arr))
        opt_thresh = float(thresholds[opt_idx]) if opt_idx < len(thresholds) else 0.5
        
        t1, t2 = st.columns([1,2])
        with t1:
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Optimal Threshold | `{opt_thresh:.3f}` |
            | Best F1 | `{f1_arr[opt_idx]:.3f}` |
            | Precision @ thresh | `{prec_arr[opt_idx]:.3f}` |
            | Recall @ thresh | `{rec_arr[opt_idx]:.3f}` |
            | Brier Score | `{brier_score_loss(y_test, y_proba):.4f}` |
            """)
        with t2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=thresholds, y=f1_arr, mode="lines", name="F1", line=dict(color="#3498db", width=2)))
            fig.add_trace(go.Scatter(x=thresholds, y=prec_arr[:-1], mode="lines", name="Precision", line=dict(color="#2ecc71", width=2, dash="dot")))
            fig.add_trace(go.Scatter(x=thresholds, y=rec_arr[:-1], mode="lines", name="Recall", line=dict(color="#e74c3c", width=2, dash="dot")))
            fig.add_vline(x=opt_thresh, line_dash="dash", line_color="orange", annotation_text=f"Opt={opt_thresh:.2f}")
            fig.update_layout(title="Threshold vs F1 / Precision / Recall", xaxis_title="Threshold", height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        y_pred = (y_proba >= opt_thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        cm_labels = ["No Purchase", "Purchase"]
        fig = go.Figure(go.Heatmap(z=cm, x=cm_labels, y=cm_labels, colorscale="Blues", text=cm,
                                   texttemplate="%{text}", textfont_size=18))
        fig.update_layout(title=f"Confusion Matrix (thr={opt_thresh:.2f})", xaxis_title="Predicted", yaxis_title="Actual", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (for tree-based or permutation)
        st.markdown(f"Feature Importance — {best_name}")
        try:
            clf = final_pipe.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                fi = clf.feature_importances_
                try:
                    feat_names = final_pipe.named_steps["prep"].get_feature_names_out()
                except:
                    feat_names = [f"f{i}" for i in range(len(fi))]
                fi_df = pd.DataFrame({"Feature": feat_names, "Importance": fi}).sort_values("Importance", ascending=True).tail(20)
                fig = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                                       marker_color=px.colors.sequential.Viridis_r[:len(fi_df)],
                                       text=fi_df["Importance"].round(4), textposition="outside"))
                fig.update_layout(title="Native Feature Importances", height=480)
                st.plotly_chart(fig, use_container_width=True)
            else:
                with st.spinner("Computing permutation importance (may take a moment)..."):
                    n_samples = min(2000, len(X_test))
                    X_sub = X_test.sample(n_samples, random_state=SEED)
                    y_sub = y_test.loc[X_sub.index]
                    perm_imp = permutation_importance(final_pipe, X_sub, y_sub, n_repeats=5, random_state=SEED,
                                                      n_jobs=-1, scoring='roc_auc')
                    fi = perm_imp.importances_mean
                    feat_names = X_test.columns
                    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": fi}).sort_values("Importance", ascending=True).tail(20)
                    fig = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                                           marker_color=px.colors.sequential.Viridis_r[:len(fi_df)],
                                           text=fi_df["Importance"].round(4), textposition="outside"))
                    fig.update_layout(title="Permutation Importance (ROC-AUC drop)", height=480)
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute feature importances: {e}")
    else:
        st.info("Configure your settings on the left and click **Train & Cross-Validate** to start.")