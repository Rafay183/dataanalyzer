"""
Data Analyzer — Streamlit app for exploring Excel / CSV datasets.

Run:
    pip install streamlit pandas plotly openpyxl
    streamlit run data_analyzer_app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data Analyzer", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, name: str, sheet=None) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if name.lower().endswith(".csv"):
        for enc in ("utf-8", "latin-1"):
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode the CSV file.")
    return pd.read_excel(buf, sheet_name=sheet or 0)


@st.cache_data(show_spinner=False)
def excel_sheets(file_bytes: bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def auto_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that look like dates into datetime."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.9 and sample.str.len().mean() >= 6:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df


def column_types(df: pd.DataFrame):
    num = df.select_dtypes(include="number").columns.tolist()
    dt = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat = [c for c in df.columns if c not in num and c not in dt]
    return num, cat, dt


def info_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Data type": df.dtypes.astype(str).values,
        "Non-null": df.notna().sum().values,
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Unique": df.nunique().values,
        "Sample value": [df[c].dropna().iloc[0] if df[c].notna().any() else None
                         for c in df.columns],
    })


def numeric_stats(df: pd.DataFrame, cols) -> pd.DataFrame:
    d = df[cols]
    stats = d.describe().T
    stats["median"] = d.median()
    stats["variance"] = d.var()
    stats["skewness"] = d.skew()
    stats["kurtosis"] = d.kurt()
    stats["IQR"] = stats["75%"] - stats["25%"]
    stats["outliers (IQR)"] = [count_outliers(d[c]) for c in cols]
    stats["missing"] = d.isna().sum()
    return stats.round(3)


def count_outliers(s: pd.Series) -> int:
    s = s.dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def fmt(x) -> str:
    if pd.isna(x):
        return "—"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if ax >= 1e3:
        return f"{x / 1e3:,.2f}K"
    return f"{x:,.2f}"


AGG = {"Sum": "sum", "Mean": "mean", "Median": "median", "Count": "count",
       "Min": "min", "Max": "max", "Std dev": "std"}


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📊 Data Analyzer")
upload = st.sidebar.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls"])

if upload is None:
    st.title("📊 Data Analyzer")
    st.info("Upload a CSV or Excel file from the sidebar to begin.")
    st.markdown(
        "**What you get:** dataset overview · column information · descriptive "
        "statistics · KPIs for numeric columns · multi-column comparison · "
        "interactive charts · cleaned-data download."
    )
    st.stop()

raw = upload.getvalue()
sheet = None
if not upload.name.lower().endswith(".csv"):
    sheets = excel_sheets(raw)
    if len(sheets) > 1:
        sheet = st.sidebar.selectbox("Sheet", sheets)

try:
    df = load_file(raw, upload.name, sheet)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the file: {e}")
    st.stop()

if st.sidebar.checkbox("Auto-detect date columns", value=True):
    df = auto_parse_dates(df)
if st.sidebar.checkbox("Drop duplicate rows", value=False):
    df = df.drop_duplicates()

num_cols, cat_cols, date_cols = column_types(df)

# Optional filter
with st.sidebar.expander("🔎 Filter data"):
    if cat_cols:
        fcol = st.selectbox("Filter by column", ["(none)"] + cat_cols)
        if fcol != "(none)":
            options = df[fcol].dropna().unique().tolist()[:500]
            chosen = st.multiselect("Values", options, default=options)
            df = df[df[fcol].isin(chosen)]
    else:
        st.caption("No categorical columns to filter on.")

st.sidebar.caption(f"Rows after filters: **{len(df):,}**")

st.title(f"📊 {upload.name}" + (f" — {sheet}" if sheet else ""))

if df.empty:
    st.warning("No rows to analyze (the file or the current filter is empty).")
    st.stop()

tabs = st.tabs(["Overview", "Column info", "Statistics", "KPIs",
                "Compare columns", "Charts", "Correlation"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    c = st.columns(6)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Numeric", len(num_cols))
    c[3].metric("Categorical", len(cat_cols))
    c[4].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c[5].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Preview")
    n = st.slider("Rows to show", 5, min(500, max(len(df), 5)), min(20, len(df)))
    st.dataframe(df.head(n), use_container_width=True)

    st.download_button("⬇️ Download current data (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="data_filtered.csv", mime="text/csv")

# ---------------------------------------------------------------- Column info
with tabs[1]:
    st.subheader("Column information")
    info = info_table(df)
    st.dataframe(info, use_container_width=True, hide_index=True)

    missing = info[info["Missing"] > 0]
    if not missing.empty:
        st.subheader("Missing values by column")
        fig = px.bar(missing.sort_values("Missing %"), x="Missing %", y="Column",
                     orientation="h", text="Missing")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values in this dataset.")

    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    st.caption(f"Memory usage: {mem:,.2f} MB")

# ---------------------------------------------------------------- Statistics
with tabs[2]:
    if num_cols:
        st.subheader("Numeric columns")
        st.dataframe(numeric_stats(df, num_cols), use_container_width=True)
    if cat_cols:
        st.subheader("Categorical columns")
        cat_desc = df[cat_cols].astype(str).describe().T
        cat_desc["top %"] = (cat_desc["freq"].astype(int) / len(df) * 100).round(2)
        st.dataframe(cat_desc, use_container_width=True)

        col = st.selectbox("Value counts for", cat_cols, key="vc")
        vc = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="Count")
        vc["Percent"] = (vc["Count"] / vc["Count"].sum() * 100).round(2)
        st.dataframe(vc, use_container_width=True, hide_index=True)
    if date_cols:
        st.subheader("Date columns")
        st.dataframe(pd.DataFrame({
            "Earliest": df[date_cols].min(),
            "Latest": df[date_cols].max(),
            "Span (days)": (df[date_cols].max() - df[date_cols].min()).dt.days,
        }), use_container_width=True)

# ---------------------------------------------------------------- KPIs
with tabs[3]:
    if not num_cols:
        st.info("No numeric columns found for KPIs.")
    else:
        sel = st.multiselect("KPI columns", num_cols, default=num_cols[:4])
        for col in sel:
            s = df[col]
            st.markdown(f"#### {col}")
            k = st.columns(6)
            k[0].metric("Total", fmt(s.sum()))
            k[1].metric("Average", fmt(s.mean()))
            k[2].metric("Median", fmt(s.median()))
            k[3].metric("Min", fmt(s.min()))
            k[4].metric("Max", fmt(s.max()))
            cv = s.std() / s.mean() * 100 if s.mean() else np.nan
            k[5].metric("Variability (CV)", "—" if pd.isna(cv) else f"{cv:.1f}%")

        if cat_cols and sel:
            st.divider()
            st.subheader("KPI by category")
            g1, g2, g3 = st.columns(3)
            gcol = g1.selectbox("Group by", cat_cols, key="kpi_g")
            vcol = g2.selectbox("Value", sel, key="kpi_v")
            agg = g3.selectbox("Aggregation", list(AGG), key="kpi_a")
            grp = (df.groupby(gcol, dropna=False)[vcol].agg(AGG[agg])
                   .sort_values(ascending=False).reset_index())
            grp["Share %"] = (grp[vcol] / grp[vcol].sum() * 100).round(2) \
                if agg in ("Sum", "Count") else np.nan
            st.dataframe(grp, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(grp.head(25), x=gcol, y=vcol,
                                   title=f"{agg} of {vcol} by {gcol}"),
                            use_container_width=True)

# ---------------------------------------------------------------- Compare
with tabs[4]:
    mode = st.radio("Comparison type",
                    ["Numeric vs numeric", "Numeric across categories",
                     "Category vs category", "Over time"], horizontal=True)

    if mode == "Numeric vs numeric":
        if len(num_cols) < 2:
            st.info("Need at least two numeric columns.")
        else:
            cols = st.multiselect("Columns to compare", num_cols, default=num_cols[:3])
            if len(cols) >= 2:
                st.dataframe(numeric_stats(df, cols).T, use_container_width=True)
                long = df[cols].melt(var_name="Column", value_name="Value")
                st.plotly_chart(px.box(long, x="Column", y="Value", points="outliers",
                                       title="Distribution comparison"),
                                use_container_width=True)
                a, b = st.columns(2)
                x = a.selectbox("X", cols, index=0)
                y = b.selectbox("Y", cols, index=1)
                color = st.selectbox("Colour by", ["(none)"] + cat_cols, key="sc_c")
                fig = px.scatter(df, x=x, y=y, opacity=0.7,
                                 color=None if color == "(none)" else color)
                st.plotly_chart(fig, use_container_width=True)
                st.metric(f"Correlation ({x} vs {y})", f"{df[x].corr(df[y]):.3f}")

    elif mode == "Numeric across categories":
        if not (num_cols and cat_cols):
            st.info("Need at least one numeric and one categorical column.")
        else:
            a, b = st.columns(2)
            g = a.selectbox("Category", cat_cols, key="nc_g")
            vals = b.multiselect("Numeric columns", num_cols, default=num_cols[:2])
            if vals:
                table = df.groupby(g, dropna=False)[vals].agg(["sum", "mean", "count"])
                st.dataframe(table.round(2), use_container_width=True)
                means = df.groupby(g, dropna=False)[vals].mean().reset_index()
                long = means.melt(id_vars=g, var_name="Metric", value_name="Average")
                st.plotly_chart(px.bar(long, x=g, y="Average", color="Metric",
                                       barmode="group", title=f"Average by {g}"),
                                use_container_width=True)

    elif mode == "Category vs category":
        if len(cat_cols) < 2:
            st.info("Need at least two categorical columns.")
        else:
            a, b, c3 = st.columns(3)
            r = a.selectbox("Rows", cat_cols, index=0)
            cc = b.selectbox("Columns", cat_cols, index=1)
            norm = c3.checkbox("Show as % of row")
            ct = pd.crosstab(df[r], df[cc], normalize="index" if norm else False)
            if norm:
                ct = (ct * 100).round(1)
            st.dataframe(ct, use_container_width=True)
            st.plotly_chart(px.imshow(ct, text_auto=True, aspect="auto",
                                      color_continuous_scale="Blues"),
                            use_container_width=True)

    else:  # Over time
        if not (date_cols and num_cols):
            st.info("Need a date column and a numeric column.")
        else:
            a, b, c3, d = st.columns(4)
            dcol = a.selectbox("Date", date_cols)
            vals = b.multiselect("Values", num_cols, default=num_cols[:2])
            fmap = {"Day": "D", "Week": "W", "Month": "MS", "Quarter": "QS", "Year": "YS"}
            freq = c3.selectbox("Period", list(fmap), index=2)
            agg = d.selectbox("Aggregation", list(AGG), key="ts_a")
            if vals:
                ts = (df.dropna(subset=[dcol]).set_index(dcol)[vals]
                      .resample(fmap[freq]).agg(AGG[agg]).reset_index())
                st.plotly_chart(px.line(ts, x=dcol, y=vals, markers=True,
                                        title=f"{agg} per {freq.lower()}"),
                                use_container_width=True)
                st.dataframe(ts, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Charts
with tabs[5]:
    chart = st.selectbox("Chart type", ["Histogram", "Box plot", "Bar chart",
                                        "Pie chart", "Line chart", "Scatter plot",
                                        "Area chart"])
    color_opt = ["(none)"] + cat_cols
    pick = lambda v: None if v == "(none)" else v  # noqa: E731

    fig = None
    if chart == "Histogram" and num_cols:
        a, b, c3 = st.columns(3)
        x = a.selectbox("Column", num_cols)
        bins = b.slider("Bins", 5, 100, 30)
        col = c3.selectbox("Colour by", color_opt)
        fig = px.histogram(df, x=x, nbins=bins, color=pick(col), marginal="box")
    elif chart == "Box plot" and num_cols:
        a, b = st.columns(2)
        y = a.selectbox("Value", num_cols)
        x = b.selectbox("Group by", color_opt)
        fig = px.box(df, x=pick(x), y=y, points="outliers")
    elif chart == "Bar chart" and cat_cols:
        a, b, c3 = st.columns(3)
        x = a.selectbox("Category", cat_cols)
        y = b.selectbox("Value", ["(count rows)"] + num_cols)
        agg = c3.selectbox("Aggregation", list(AGG))
        top = st.slider("Top N", 3, 50, 15)
        if y == "(count rows)":
            data = df[x].value_counts().head(top).rename_axis(x).reset_index(name="Count")
            fig = px.bar(data, x=x, y="Count", text_auto=True)
        else:
            data = (df.groupby(x)[y].agg(AGG[agg]).sort_values(ascending=False)
                    .head(top).reset_index())
            fig = px.bar(data, x=x, y=y, text_auto=".2s")
    elif chart == "Pie chart" and cat_cols:
        a, b = st.columns(2)
        names = a.selectbox("Category", cat_cols)
        val = b.selectbox("Value", ["(count rows)"] + num_cols)
        if val == "(count rows)":
            data = df[names].value_counts().head(10).rename_axis(names).reset_index(name="Count")
            fig = px.pie(data, names=names, values="Count", hole=0.4)
        else:
            data = df.groupby(names)[val].sum().nlargest(10).reset_index()
            fig = px.pie(data, names=names, values=val, hole=0.4)
    elif chart in ("Line chart", "Area chart") and num_cols:
        a, b = st.columns(2)
        x = a.selectbox("X axis", date_cols + num_cols + cat_cols)
        ys = b.multiselect("Y axis", num_cols, default=num_cols[:1])
        if ys:
            data = df.sort_values(x)
            fn = px.line if chart == "Line chart" else px.area
            fig = fn(data, x=x, y=ys)
    elif chart == "Scatter plot" and len(num_cols) >= 2:
        a, b, c3, d = st.columns(4)
        x = a.selectbox("X", num_cols, index=0)
        y = b.selectbox("Y", num_cols, index=1)
        col = c3.selectbox("Colour by", color_opt)
        size = d.selectbox("Size by", ["(none)"] + num_cols)
        data = df.dropna(subset=[c for c in [x, y, pick(size)] if c])
        if pick(size):
            data = data[data[size] >= 0]
        fig = px.scatter(data, x=x, y=y, color=pick(col), size=pick(size), opacity=0.7)

    if fig is not None:
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("This dataset doesn't have the column types this chart needs.")

# ---------------------------------------------------------------- Correlation
with tabs[6]:
    if len(num_cols) < 2:
        st.info("Need at least two numeric columns.")
    else:
        method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
        corr = df[num_cols].corr(method=method)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                                  color_continuous_scale="RdBu_r", aspect="auto"),
                        use_container_width=True)
        pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
                 .stack().rename("Correlation").reset_index())
        pairs.columns = ["Column A", "Column B", "Correlation"]
        pairs = pairs.reindex(pairs["Correlation"].abs().sort_values(ascending=False).index)
        st.subheader("Strongest relationships")
        st.dataframe(pairs.head(15).round(3), use_container_width=True, hide_index=True)
