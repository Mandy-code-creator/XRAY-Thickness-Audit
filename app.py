
import io
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="XRAY Coating Thickness Audit",
    page_icon="📏",
    layout="wide",
)

st.title("XRAY Coating Thickness Audit")
st.success("Version 2026-07-31 — Formula: North/Center/South = Top + Back (NO division by 2)")
st.caption(
    "Analyze coating thickness by Coating Type → Upper Coating "
    "and compare each coil with Target and Lower Limit."
)


# =========================================================
# 2. REQUIRED COLUMNS
# =========================================================
REQUIRED_COLUMNS = [
    "鍍製別",
    "上鍍層",
    "訂單號碼",
    "產出鋼捲號碼",
    "鍍層目標值",
    "鍍層下限值",
    "XRAY_A_T_N",
    "XRAY_A_T_C",
    "XRAY_A_T_S",
    "XRAY_A_B_N",
    "XRAY_A_B_C",
    "XRAY_A_B_S",
]

NUMERIC_COLUMNS = [
    "鍍層目標值",
    "鍍層下限值",
    "XRAY_A_T_N",
    "XRAY_A_T_C",
    "XRAY_A_T_S",
    "XRAY_A_B_N",
    "XRAY_A_B_C",
    "XRAY_A_B_S",
]

TEXT_COLUMNS = [
    "鍍製別",
    "上鍍層",
    "訂單號碼",
    "產出鋼捲號碼",
]


# =========================================================
# 3. FILE READING
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u3000", " ", regex=False)
        .str.strip()
    )
    return result


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        raw_bytes = uploaded_file.getvalue()
        encodings = ["utf-8-sig", "utf-8", "big5", "cp950"]
        last_error = None

        for encoding in encodings:
            try:
                return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
            except Exception as exc:
                last_error = exc

        raise ValueError(f"Unable to read CSV file: {last_error}")

    if filename.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Supported formats: CSV, XLSX, XLSM and XLS.")


# =========================================================
# 4. DATA VALIDATION
# =========================================================
def validate_required_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("，", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Correct calculation logic:

    North Thickness = XRAY_A_T_N + XRAY_A_B_N
    Center Thickness = XRAY_A_T_C + XRAY_A_B_C
    South Thickness = XRAY_A_T_S + XRAY_A_B_S

    Coil Average Thickness =
        (North Thickness + Center Thickness + South Thickness) / 3
    """
    data = normalize_columns(df)

    for column in TEXT_COLUMNS:
        data[column] = data[column].astype("string").str.strip()

    for column in NUMERIC_COLUMNS:
        data[column] = clean_numeric_series(data[column])

    reject_reasons = []

    for _, row in data.iterrows():
        reasons = []

        for column in ["鍍製別", "上鍍層", "產出鋼捲號碼"]:
            value = row[column]
            if pd.isna(value) or str(value).strip() == "":
                reasons.append(f"Missing {column}")

        for column in NUMERIC_COLUMNS:
            if pd.isna(row[column]):
                reasons.append(f"Invalid or missing {column}")

        # Target and lower limit equal to zero are not valid for comparison.
        if pd.notna(row["鍍層目標值"]) and row["鍍層目標值"] <= 0:
            reasons.append("Target must be greater than 0")

        if pd.notna(row["鍍層下限值"]) and row["鍍層下限值"] <= 0:
            reasons.append("Lower Limit must be greater than 0")

        if (
            pd.notna(row["鍍層目標值"])
            and pd.notna(row["鍍層下限值"])
            and row["鍍層下限值"] > row["鍍層目標值"]
        ):
            reasons.append("Lower Limit is greater than Target")

        reject_reasons.append("; ".join(reasons))

    data["Reject Reason"] = reject_reasons

    rejected = data[data["Reject Reason"] != ""].copy()
    valid = data[data["Reject Reason"] == ""].copy()

    if valid.empty:
        return valid, rejected

    # -----------------------------------------------------
    # Correct coating thickness calculation: Top + Back
    # -----------------------------------------------------
    valid["North Thickness"] = (
        valid["XRAY_A_T_N"] + valid["XRAY_A_B_N"]
    )

    valid["Center Thickness"] = (
        valid["XRAY_A_T_C"] + valid["XRAY_A_B_C"]
    )

    valid["South Thickness"] = (
        valid["XRAY_A_T_S"] + valid["XRAY_A_B_S"]
    )

    position_columns = [
        "North Thickness",
        "Center Thickness",
        "South Thickness",
    ]

    valid["Coil Average Thickness"] = valid[position_columns].mean(axis=1)

    # Difference from target
    valid["Target Deviation"] = (
        valid["Coil Average Thickness"] - valid["鍍層目標值"]
    )

    valid["Absolute Target Deviation"] = valid["Target Deviation"].abs()

    valid["Target Deviation (%)"] = (
        valid["Target Deviation"] / valid["鍍層目標值"] * 100
    )

    # Margin from lower limit
    valid["Lower Limit Margin"] = (
        valid["Coil Average Thickness"] - valid["鍍層下限值"]
    )

    valid["Lower Limit Margin (%)"] = (
        valid["Lower Limit Margin"] / valid["鍍層下限值"] * 100
    )

    # Position-level comparison
    for position in ["North", "Center", "South"]:
        thickness_column = f"{position} Thickness"
        valid[f"{position} Target Deviation"] = (
            valid[thickness_column] - valid["鍍層目標值"]
        )
        valid[f"{position} Lower Limit Margin"] = (
            valid[thickness_column] - valid["鍍層下限值"]
        )

    # Cross-width uniformity
    valid["Minimum Position Thickness"] = valid[position_columns].min(axis=1)
    valid["Maximum Position Thickness"] = valid[position_columns].max(axis=1)

    valid["Cross-Width Range"] = (
        valid["Maximum Position Thickness"]
        - valid["Minimum Position Thickness"]
    )

    valid["Cross-Width Range (%)"] = (
        valid["Cross-Width Range"]
        / valid["Coil Average Thickness"]
        * 100
    )

    min_position_column = valid[position_columns].idxmin(axis=1)
    valid["Minimum Position"] = min_position_column.map(
        {
            "North Thickness": "North",
            "Center Thickness": "Center",
            "South Thickness": "South",
        }
    )

    # Status logic
    valid["Any Position Below Lower Limit"] = (
        valid["Minimum Position Thickness"] < valid["鍍層下限值"]
    )

    valid["Average Below Lower Limit"] = (
        valid["Coil Average Thickness"] < valid["鍍層下限值"]
    )

    valid["Average Below Target"] = (
        valid["Coil Average Thickness"] < valid["鍍層目標值"]
    )

    valid["Coil Status"] = np.select(
        [
            valid["Any Position Below Lower Limit"],
            valid["Average Below Target"],
        ],
        [
            "Any Position Below Lower Limit",
            "Average Below Target",
        ],
        default="Meets Requirement",
    )

    return valid, rejected


# =========================================================
# 5. OPTIONAL COIL-LEVEL AGGREGATION
# =========================================================
def aggregate_to_one_row_per_coil(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the same output coil appears multiple times, aggregate raw XRAY
    measurements before recalculating all indicators.
    """
    if df.empty:
        return df.copy()

    group_columns = [
        "鍍製別",
        "上鍍層",
        "訂單號碼",
        "產出鋼捲號碼",
    ]

    aggregation = {
        "鍍層目標值": "median",
        "鍍層下限值": "median",
        "XRAY_A_T_N": "mean",
        "XRAY_A_T_C": "mean",
        "XRAY_A_T_S": "mean",
        "XRAY_A_B_N": "mean",
        "XRAY_A_B_C": "mean",
        "XRAY_A_B_S": "mean",
    }

    grouped = (
        df.groupby(group_columns, dropna=False, as_index=False)
        .agg(aggregation)
    )

    recalculated, _ = prepare_data(grouped)
    return recalculated


# =========================================================
# 6. SUMMARY TABLE
# =========================================================
def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    working = df.copy()

    working["Below Lower Limit Flag"] = (
        working["Any Position Below Lower Limit"].astype(int)
    )
    working["Below Target Flag"] = working["Average Below Target"].astype(int)
    working["At or Above Target Flag"] = (
        working["Coil Average Thickness"] >= working["鍍層目標值"]
    ).astype(int)

    summary = (
        working.groupby(["鍍製別", "上鍍層"], dropna=False)
        .agg(
            Coil_Count=("產出鋼捲號碼", "nunique"),
            Average_Thickness=("Coil Average Thickness", "mean"),
            Average_Target_Deviation=("Target Deviation", "mean"),
            Median_Target_Deviation=("Target Deviation", "median"),
            Average_Lower_Limit_Margin=("Lower Limit Margin", "mean"),
            Minimum_Lower_Limit_Margin=("Lower Limit Margin", "min"),
            Coils_With_Position_Below_Limit=("Below Lower Limit Flag", "sum"),
            Coils_Below_Target=("Below Target Flag", "sum"),
            Coils_At_Or_Above_Target=("At or Above Target Flag", "sum"),
            Average_Cross_Width_Range=("Cross-Width Range", "mean"),
            Average_Cross_Width_Range_Percent=("Cross-Width Range (%)", "mean"),
        )
        .reset_index()
    )

    summary["Position Below Limit Rate (%)"] = (
        summary["Coils_With_Position_Below_Limit"]
        / summary["Coil_Count"]
        * 100
    )

    summary["Below Target Rate (%)"] = (
        summary["Coils_Below_Target"]
        / summary["Coil_Count"]
        * 100
    )

    summary["At or Above Target Rate (%)"] = (
        summary["Coils_At_Or_Above_Target"]
        / summary["Coil_Count"]
        * 100
    )

    return summary


# =========================================================
# 7. EXPORT
# =========================================================
def dataframe_to_excel_bytes(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        detail_df.to_excel(writer, index=False, sheet_name="Coil Detail")
        summary_df.to_excel(writer, index=False, sheet_name="Group Summary")
        rejected_df.to_excel(writer, index=False, sheet_name="Rejected Data")

    output.seek(0)
    return output.getvalue()


def style_dataframe(df: pd.DataFrame, decimals: int = 2):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    format_map = {
        column: f"{{:.{decimals}f}}" for column in numeric_columns
    }
    return df.style.format(format_map)


# =========================================================
# 8. CHART FUNCTIONS
# =========================================================
def plot_coil_average_vs_limits(df: pd.DataFrame):
    chart_df = df.reset_index(drop=True).copy()
    x = np.arange(len(chart_df))

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        x,
        chart_df["Coil Average Thickness"],
        marker="o",
        linewidth=1.5,
        label="Coil Average",
    )
    ax.plot(
        x,
        chart_df["鍍層目標值"],
        linestyle="--",
        linewidth=1.5,
        label="Target",
    )
    ax.plot(
        x,
        chart_df["鍍層下限值"],
        linestyle=":",
        linewidth=1.8,
        label="Lower Limit",
    )

    ax.set_title("Coil Average vs Target and Lower Limit", pad=16)
    ax.set_xlabel("Output Coil Number")
    ax.set_ylabel("Coating Thickness")
    ax.grid(True, alpha=0.3)
    ax.legend()

    tick_step = max(1, len(chart_df) // 20)
    tick_positions = x[::tick_step]
    tick_labels = (
        chart_df["產出鋼捲號碼"]
        .astype(str)
        .iloc[::tick_step]
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=60, ha="right")

    for spine in ax.spines.values():
        spine.set_visible(True)

    fig.tight_layout()
    return fig


def select_top_deviation_data(
    df: pd.DataFrame,
    ranking_method: str,
    top_n: int,
) -> Tuple[pd.DataFrame, str, str]:
    working = df.copy()

    if ranking_method == "Absolute Target Deviation":
        ranked = working.nlargest(top_n, "Absolute Target Deviation")
        value_column = "Target Deviation"
        title = f"Top {top_n} Coils by Absolute Target Deviation"

    elif ranking_method == "Above-Target Deviation":
        ranked = working.nlargest(top_n, "Target Deviation")
        value_column = "Target Deviation"
        title = f"Top {top_n} Coils Above Target"

    elif ranking_method == "Below-Target Deviation":
        ranked = working.nsmallest(top_n, "Target Deviation")
        value_column = "Target Deviation"
        title = f"Top {top_n} Coils Below Target"

    elif ranking_method == "Lower-Limit Risk":
        ranked = working.nsmallest(top_n, "Lower Limit Margin")
        value_column = "Lower Limit Margin"
        title = f"Top {top_n} Coils Closest to or Below Lower Limit"

    else:
        ranked = working.nlargest(top_n, "Cross-Width Range")
        value_column = "Cross-Width Range"
        title = f"Top {top_n} Coils by Cross-Width Variation"

    return ranked.copy(), value_column, title


def plot_top_deviation(
    ranked_df: pd.DataFrame,
    value_column: str,
    title: str,
):
    chart_df = ranked_df.sort_values(value_column, ascending=True).copy()

    labels = (
        chart_df["產出鋼捲號碼"].astype(str)
        + " | "
        + chart_df["鍍製別"].astype(str)
        + " | "
        + chart_df["上鍍層"].astype(str)
    )

    fig_height = max(6, len(chart_df) * 0.38)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    bars = ax.barh(labels, chart_df[value_column])
    ax.axvline(0, linewidth=1.0)

    ax.set_title(title, pad=16)
    ax.set_xlabel(value_column)
    ax.set_ylabel("Output Coil Number | Coating Type | Upper Coating")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, value in zip(bars, chart_df[value_column]):
        offset = max(abs(chart_df[value_column]).max() * 0.01, 0.1)
        if value >= 0:
            x_position = value + offset
            horizontal_alignment = "left"
        else:
            x_position = value - offset
            horizontal_alignment = "right"

        ax.text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=8,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)

    fig.tight_layout()
    return fig


def plot_position_profile(df: pd.DataFrame):
    chart_df = df.reset_index(drop=True).copy()
    x = np.arange(len(chart_df))

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(x, chart_df["North Thickness"], marker="o", label="North")
    ax.plot(x, chart_df["Center Thickness"], marker="o", label="Center")
    ax.plot(x, chart_df["South Thickness"], marker="o", label="South")
    ax.plot(
        x,
        chart_df["鍍層下限值"],
        linestyle=":",
        linewidth=1.8,
        label="Lower Limit",
    )

    ax.set_title("North / Center / South Thickness Profile", pad=16)
    ax.set_xlabel("Output Coil Number")
    ax.set_ylabel("Coating Thickness")
    ax.grid(True, alpha=0.3)
    ax.legend()

    tick_step = max(1, len(chart_df) // 20)
    tick_positions = x[::tick_step]
    tick_labels = (
        chart_df["產出鋼捲號碼"]
        .astype(str)
        .iloc[::tick_step]
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=60, ha="right")

    for spine in ax.spines.values():
        spine.set_visible(True)

    fig.tight_layout()
    return fig


# =========================================================
# 9. UPLOAD DATA
# =========================================================
st.sidebar.header("Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload FRPMES0131_CGL file",
    type=["csv", "xlsx", "xlsm", "xls"],
    help="Supported formats: CSV, XLSX, XLSM and XLS.",
)

if uploaded_file is None:
    st.info("Upload the FRPMES0131_CGL data file from the sidebar.")
    st.stop()

try:
    raw_df = read_uploaded_file(uploaded_file)
except Exception as exc:
    st.error(f"File reading failed: {exc}")
    st.stop()

raw_df = normalize_columns(raw_df)

missing_columns = validate_required_columns(raw_df)

if missing_columns:
    st.error("The uploaded file is missing required columns:")
    st.code("\n".join(missing_columns))

    with st.expander("Columns found in the uploaded file"):
        st.write(list(raw_df.columns))

    st.stop()


# =========================================================
# 10. PREPARE DATA
# =========================================================
valid_df, rejected_df = prepare_data(raw_df)

if valid_df.empty:
    st.error(
        "No valid rows are available for analysis. "
        "Check Target, Lower Limit and XRAY values."
    )

    if not rejected_df.empty:
        st.dataframe(rejected_df, use_container_width=True)

    st.stop()


# =========================================================
# 11. ANALYSIS SETTINGS
# =========================================================
st.sidebar.header("Analysis Settings")

data_level = st.sidebar.radio(
    "Data Level",
    ["One Row per Coil", "Original Rows"],
    index=0,
    help=(
        "One Row per Coil aggregates repeated records of the same coil "
        "before recalculating thickness."
    ),
)

if data_level == "One Row per Coil":
    analysis_df = aggregate_to_one_row_per_coil(valid_df)
else:
    analysis_df = valid_df.copy()


# =========================================================
# 12. FILTERS
# =========================================================
st.sidebar.header("Filters")

coating_type_options = sorted(
    analysis_df["鍍製別"].dropna().astype(str).unique().tolist()
)

selected_coating_types = st.sidebar.multiselect(
    "Coating Type",
    options=coating_type_options,
    default=coating_type_options,
)

filtered_df = analysis_df[
    analysis_df["鍍製別"].astype(str).isin(selected_coating_types)
].copy()

upper_coating_options = sorted(
    filtered_df["上鍍層"].dropna().astype(str).unique().tolist()
)

selected_upper_coatings = st.sidebar.multiselect(
    "Upper Coating",
    options=upper_coating_options,
    default=upper_coating_options,
)

filtered_df = filtered_df[
    filtered_df["上鍍層"].astype(str).isin(selected_upper_coatings)
].copy()

order_options = sorted(
    filtered_df["訂單號碼"].dropna().astype(str).unique().tolist()
)

selected_orders = st.sidebar.multiselect(
    "Order Number",
    options=order_options,
    default=[],
    help="Leave empty to include all orders.",
)

if selected_orders:
    filtered_df = filtered_df[
        filtered_df["訂單號碼"].astype(str).isin(selected_orders)
    ].copy()

coil_keyword = st.sidebar.text_input(
    "Search Output Coil Number",
    value="",
)

if coil_keyword.strip():
    filtered_df = filtered_df[
        filtered_df["產出鋼捲號碼"]
        .astype(str)
        .str.contains(coil_keyword.strip(), case=False, na=False)
    ].copy()

filtered_df = filtered_df.sort_values(
    ["鍍製別", "上鍍層", "訂單號碼", "產出鋼捲號碼"]
).reset_index(drop=True)

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()


# =========================================================
# 13. KPI
# =========================================================
summary_df = build_group_summary(filtered_df)

coil_count = filtered_df["產出鋼捲號碼"].nunique()
below_limit_count = int(filtered_df["Any Position Below Lower Limit"].sum())
below_target_count = int(filtered_df["Average Below Target"].sum())
average_target_deviation = filtered_df["Target Deviation"].mean()
average_lower_margin = filtered_df["Lower Limit Margin"].mean()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("Coil Count", f"{coil_count:,}")

kpi2.metric(
    "Any Position Below Limit",
    f"{below_limit_count:,}",
    f"{below_limit_count / len(filtered_df) * 100:.1f}%",
    delta_color="inverse",
)

kpi3.metric(
    "Average Below Target",
    f"{below_target_count:,}",
    f"{below_target_count / len(filtered_df) * 100:.1f}%",
    delta_color="inverse",
)

kpi4.metric(
    "Average Target Deviation",
    f"{average_target_deviation:.2f}",
)

kpi5.metric(
    "Average Lower-Limit Margin",
    f"{average_lower_margin:.2f}",
)


# =========================================================
# 14. TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Management Summary",
        "Coil Detail",
        "Charts",
        "Data Quality",
    ]
)


# =========================================================
# TAB 1: MANAGEMENT SUMMARY
# =========================================================
with tab1:
    st.subheader("Summary by Coating Type → Upper Coating")

    summary_columns = [
        "鍍製別",
        "上鍍層",
        "Coil_Count",
        "Average_Thickness",
        "Average_Target_Deviation",
        "Median_Target_Deviation",
        "Average_Lower_Limit_Margin",
        "Minimum_Lower_Limit_Margin",
        "Coils_With_Position_Below_Limit",
        "Position Below Limit Rate (%)",
        "Coils_Below_Target",
        "Below Target Rate (%)",
        "Coils_At_Or_Above_Target",
        "At or Above Target Rate (%)",
        "Average_Cross_Width_Range",
        "Average_Cross_Width_Range_Percent",
    ]

    st.dataframe(
        style_dataframe(summary_df[summary_columns], 2),
        use_container_width=True,
        height=480,
    )

    st.markdown(
        """
        **Metric definitions**

        - **Target Deviation** = Coil Average Thickness − Target.
        - **Lower-Limit Margin** = Coil Average Thickness − Lower Limit.
        - **Any Position Below Limit** means North, Center or South is below the Lower Limit.
        - **Cross-Width Range** = Maximum position thickness − Minimum position thickness.
        """
    )


# =========================================================
# TAB 2: COIL DETAIL
# =========================================================
with tab2:
    st.subheader("Coil-Level Analysis")

    detail_columns = [
        "鍍製別",
        "上鍍層",
        "訂單號碼",
        "產出鋼捲號碼",
        "鍍層目標值",
        "鍍層下限值",
        "XRAY_A_T_N",
        "XRAY_A_B_N",
        "North Thickness",
        "XRAY_A_T_C",
        "XRAY_A_B_C",
        "Center Thickness",
        "XRAY_A_T_S",
        "XRAY_A_B_S",
        "South Thickness",
        "Coil Average Thickness",
        "Target Deviation",
        "Target Deviation (%)",
        "Lower Limit Margin",
        "Lower Limit Margin (%)",
        "Minimum Position",
        "Minimum Position Thickness",
        "Cross-Width Range",
        "Cross-Width Range (%)",
        "Coil Status",
    ]

    st.dataframe(
        style_dataframe(filtered_df[detail_columns], 2),
        use_container_width=True,
        height=620,
    )

    abnormal_df = filtered_df[
        filtered_df["Coil Status"] != "Meets Requirement"
    ].copy()

    st.subheader("Abnormal Coil List")

    if abnormal_df.empty:
        st.success("No abnormal coils were found under the current filters.")
    else:
        st.dataframe(
            style_dataframe(abnormal_df[detail_columns], 2),
            use_container_width=True,
            height=400,
        )

    export_bytes = dataframe_to_excel_bytes(
        detail_df=filtered_df[detail_columns],
        summary_df=summary_df[summary_columns],
        rejected_df=rejected_df,
    )

    st.download_button(
        "Download Analysis Excel",
        data=export_bytes,
        file_name="XRAY_Coating_Thickness_Analysis.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )


# =========================================================
# TAB 3: CHARTS
# =========================================================
with tab3:
    st.subheader("Visual Analysis")

    maximum_chart_rows = min(300, len(filtered_df))

    rows_to_show = st.slider(
        "Number of coils shown in trend charts",
        min_value=1,
        max_value=max(1, maximum_chart_rows),
        value=min(50, maximum_chart_rows),
        step=1,
    )

    trend_df = filtered_df.head(rows_to_show).copy()

    st.pyplot(
        plot_coil_average_vs_limits(trend_df),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Top Deviation Ranking")

    ranking_method = st.selectbox(
        "Ranking Method",
        [
            "Absolute Target Deviation",
            "Above-Target Deviation",
            "Below-Target Deviation",
            "Lower-Limit Risk",
            "Cross-Width Variation",
        ],
        index=0,
    )

    top_n = st.slider(
        "Top N",
        min_value=5,
        max_value=min(50, len(filtered_df)),
        value=min(20, len(filtered_df)),
        step=1,
    )

    ranked_df, ranking_value_column, ranking_title = (
        select_top_deviation_data(
            filtered_df,
            ranking_method,
            top_n,
        )
    )

    st.pyplot(
        plot_top_deviation(
            ranked_df,
            ranking_value_column,
            ranking_title,
        ),
        use_container_width=True,
    )

    ranking_table_columns = [
        "鍍製別",
        "上鍍層",
        "訂單號碼",
        "產出鋼捲號碼",
        "鍍層目標值",
        "鍍層下限值",
        "North Thickness",
        "Center Thickness",
        "South Thickness",
        "Coil Average Thickness",
        "Target Deviation",
        "Absolute Target Deviation",
        "Lower Limit Margin",
        "Cross-Width Range",
        "Coil Status",
    ]

    st.dataframe(
        style_dataframe(ranked_df[ranking_table_columns], 2),
        use_container_width=True,
        height=500,
    )

    st.divider()

    st.pyplot(
        plot_position_profile(trend_df),
        use_container_width=True,
    )


# =========================================================
# TAB 4: DATA QUALITY
# =========================================================
with tab4:
    st.subheader("Data Quality")

    quality1, quality2, quality3 = st.columns(3)

    quality1.metric("Original Rows", f"{len(raw_df):,}")
    quality2.metric("Valid Rows", f"{len(valid_df):,}")
    quality3.metric("Rejected Rows", f"{len(rejected_df):,}")

    st.markdown(
        """
        Rows are rejected when required values are missing, XRAY values are
        nonnumeric, Target or Lower Limit is zero or negative, or Lower Limit
        is greater than Target.
        """
    )

    if rejected_df.empty:
        st.success("No rejected rows.")
    else:
        rejected_columns = [
            "鍍製別",
            "上鍍層",
            "訂單號碼",
            "產出鋼捲號碼",
            "鍍層目標值",
            "鍍層下限值",
            "XRAY_A_T_N",
            "XRAY_A_T_C",
            "XRAY_A_T_S",
            "XRAY_A_B_N",
            "XRAY_A_B_C",
            "XRAY_A_B_S",
            "Reject Reason",
        ]

        st.dataframe(
            rejected_df[rejected_columns],
            use_container_width=True,
            height=550,
        )


# =========================================================
# 15. FOOTNOTE
# =========================================================
st.caption(
    "Calculation: "
    "North = XRAY_A_T_N + XRAY_A_B_N; "
    "Center = XRAY_A_T_C + XRAY_A_B_C; "
    "South = XRAY_A_T_S + XRAY_A_B_S; "
    "Coil Average = (North + Center + South) / 3."
)
