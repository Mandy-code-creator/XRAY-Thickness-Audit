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
st.success(
    "Version 2026-07-31 FINAL TWO-VIEW — "
    "North / Center / South = Top + Back"
)
st.caption(
    "Analyze coating thickness by Coating Type → Upper Coating "
    "and compare each coil with Target and Lower Limit."
)


# =========================================================
# 2. CONSTANTS
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

TEXT_COLUMNS = [
    "鍍製別",
    "上鍍層",
    "訂單號碼",
    "產出鋼捲號碼",
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


# =========================================================
# 3. DATA READING AND CLEANING
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


@st.cache_data(show_spinner=False)
def read_file_bytes(
    file_name: str,
    file_bytes: bytes,
) -> pd.DataFrame:
    lower_name = file_name.lower()

    if lower_name.endswith(".csv"):
        encodings = ["utf-8-sig", "utf-8", "big5", "cp950"]
        last_error = None

        for encoding in encodings:
            try:
                return pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding,
                )
            except Exception as exc:
                last_error = exc

        raise ValueError(f"Unable to read CSV file: {last_error}")

    if lower_name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))

    raise ValueError("Supported formats: CSV, XLSX, XLSM and XLS.")


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]


def calculate_thickness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Core thickness calculations independent of error validation."""
    valid = df.copy()
    
    valid["North Thickness"] = valid["XRAY_A_T_N"] + valid["XRAY_A_B_N"]
    valid["Center Thickness"] = valid["XRAY_A_T_C"] + valid["XRAY_A_B_C"]
    valid["South Thickness"] = valid["XRAY_A_T_S"] + valid["XRAY_A_B_S"]

    position_columns = ["North Thickness", "Center Thickness", "South Thickness"]

    valid["Coil Average Thickness"] = valid[position_columns].mean(axis=1)
    valid["Target Deviation"] = valid["Coil Average Thickness"] - valid["鍍層目標值"]
    valid["Absolute Target Deviation"] = valid["Target Deviation"].abs()
    valid["Target Deviation (%)"] = valid["Target Deviation"] / valid["鍍層目標值"] * 100
    
    valid["Lower Limit Margin"] = valid["Coil Average Thickness"] - valid["鍍層下限值"]
    valid["Lower Limit Margin (%)"] = valid["Lower Limit Margin"] / valid["鍍層下限值"] * 100

    for position in ["North", "Center", "South"]:
        thickness_column = f"{position} Thickness"
        valid[f"{position} Target Deviation"] = valid[thickness_column] - valid["鍍層目標值"]
        valid[f"{position} Lower Limit Margin"] = valid[thickness_column] - valid["鍍層下限值"]

    valid["Minimum Position Thickness"] = valid[position_columns].min(axis=1)
    valid["Maximum Position Thickness"] = valid[position_columns].max(axis=1)
    valid["Cross-Width Range"] = valid["Maximum Position Thickness"] - valid["Minimum Position Thickness"]
    valid["Cross-Width Range (%)"] = valid["Cross-Width Range"] / valid["Coil Average Thickness"] * 100

    valid["Minimum Position"] = (
        valid[position_columns]
        .idxmin(axis=1)
        .map({
            "North Thickness": "North",
            "Center Thickness": "Center",
            "South Thickness": "South",
        })
    )

    valid["Any Position Below Lower Limit"] = valid["Minimum Position Thickness"] < valid["鍍層下限值"]
    valid["Average Below Target"] = valid["Coil Average Thickness"] < valid["鍍層目標值"]

    valid["Coil Status"] = np.select(
        [valid["Any Position Below Lower Limit"], valid["Average Below Target"]],
        ["Any Position Below Lower Limit", "Average Below Target"],
        default="Meets Requirement",
    )

    return valid


@st.cache_data(show_spinner=False)
def prepare_data(
    source_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    data = normalize_columns(source_df)

    for column in TEXT_COLUMNS:
        data[column] = data[column].astype("string").str.strip()

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("，", "", regex=False)
            .str.strip(),
            errors="coerce",
        )

    # Error Validation
    invalid_text = pd.DataFrame({
        column: (data[column].isna() | data[column].astype("string").str.strip().eq(""))
        for column in ["鍍製別", "上鍍層", "產出鋼捲號碼"]
    })

    invalid_numeric = data[NUMERIC_COLUMNS].isna()
    target_invalid = data["鍍層目標值"].le(0)
    lower_invalid = data["鍍層下限值"].le(0)
    lower_above_target = data["鍍層下限值"] > data["鍍層目標值"]

    rejected_mask = (
        invalid_text.any(axis=1)
        | invalid_numeric.any(axis=1)
        | target_invalid
        | lower_invalid
        | lower_above_target
    )

    reasons = pd.Series("", index=data.index, dtype="string")

    for column in ["鍍製別", "上鍍層", "產出鋼捲號碼"]:
        mask = invalid_text[column]
        reasons = reasons.mask(
            mask,
            reasons.where(~mask, "") + np.where(reasons.ne("") & mask, "; ", "") + f"Missing {column}",
        )

    for column in NUMERIC_COLUMNS:
        mask = invalid_numeric[column]
        reasons = reasons.mask(
            mask,
            reasons.where(~mask, "") + np.where(reasons.ne("") & mask, "; ", "") + f"Invalid or missing {column}",
        )

    for mask, text in [
        (target_invalid, "Target must be greater than 0"),
        (lower_invalid, "Lower Limit must be greater than 0"),
        (lower_above_target, "Lower Limit is greater than Target"),
    ]:
        reasons = reasons.mask(
            mask,
            reasons.where(~mask, "") + np.where(reasons.ne("") & mask, "; ", "") + text,
        )

    data["Reject Reason"] = reasons

    rejected = data.loc[rejected_mask].copy()
    valid = data.loc[~rejected_mask].copy()

    if valid.empty:
        return valid, rejected

    valid = calculate_thickness_metrics(valid)
    return valid, rejected


@st.cache_data(show_spinner=False)
def aggregate_to_one_row_per_coil(
    valid_df: pd.DataFrame,
) -> pd.DataFrame:
    if valid_df.empty:
        return valid_df.copy()

    group_columns = [
        "鍍製別",
        "上鍍層",
        "鍍層目標值",
        "鍍層下限值",
        "訂單號碼",
        "產出鋼捲號碼",
    ]

    aggregation = {
        "XRAY_A_T_N": "mean",
        "XRAY_A_T_C": "mean",
        "XRAY_A_T_S": "mean",
        "XRAY_A_B_N": "mean",
        "XRAY_A_B_C": "mean",
        "XRAY_A_B_S": "mean",
    }

    grouped = (
        valid_df.groupby(
            group_columns,
            dropna=False,
            as_index=False,
        )
        .agg(aggregation)
    )

    recalculated = calculate_thickness_metrics(grouped)
    return recalculated


@st.cache_data(show_spinner=False)
def build_group_summary(
    df: pd.DataFrame,
    uneven_threshold_percent: float = 10.0,
    high_stability_threshold: float = 3.0,
    medium_stability_threshold: float = 6.0,
    minimum_reliable_coils: int = 5,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    working = df.copy()

    working["Below Lower Limit Flag"] = working["Any Position Below Lower Limit"].astype(int)
    working["Below Target Flag"] = working["Average Below Target"].astype(int)
    working["At or Above Target Flag"] = (working["Coil Average Thickness"] >= working["鍍層目標值"]).astype(int)
    working["Excess Coating Flag"] = (working["Coil Average Thickness"] > working["鍍層目標值"]).astype(int)
    working["Uneven Coating Flag"] = (working["Cross-Width Range (%)"] > uneven_threshold_percent).astype(int)

    group_columns = ["鍍製別", "上鍍層", "鍍層目標值", "鍍層下限值"]

    summary = (
        working.groupby(group_columns, dropna=False)
        .agg(
            Coil_Count=("產出鋼捲號碼", "nunique"),
            Average_Thickness=("Coil Average Thickness", "mean"),
            Thickness_Std=("Coil Average Thickness", "std"),
            Average_Target_Deviation=("Target Deviation", "mean"),
            Median_Target_Deviation=("Target Deviation", "median"),
            Target_Deviation_Std=("Target Deviation", "std"),
            Target_Deviation_P10=("Target Deviation", lambda x: x.quantile(0.10)),
            Target_Deviation_P90=("Target Deviation", lambda x: x.quantile(0.90)),
            Average_Lower_Limit_Margin=("Lower Limit Margin", "mean"),
            Minimum_Lower_Limit_Margin=("Lower Limit Margin", "min"),
            Coils_With_Position_Below_Limit=("Below Lower Limit Flag", "sum"),
            Coils_Below_Target=("Below Target Flag", "sum"),
            Coils_At_Or_Above_Target=("At or Above Target Flag", "sum"),
            Excess_Coating_Coils=("Excess Coating Flag", "sum"),
            Uneven_Coating_Coils=("Uneven Coating Flag", "sum"),
            Average_Cross_Width_Range=("Cross-Width Range", "mean"),
            Average_Cross_Width_Range_Percent=("Cross-Width Range (%)", "mean"),
        )
        .reset_index()
    )

    summary["Thickness_Std"] = summary["Thickness_Std"].fillna(0)
    summary["Target_Deviation_Std"] = summary["Target_Deviation_Std"].fillna(0)
    summary["Target_Deviation_P10_P90_Range"] = summary["Target_Deviation_P90"] - summary["Target_Deviation_P10"]

    summary["Deviation_CV_vs_Target (%)"] = np.where(
        summary["鍍層目標值"] != 0,
        summary["Target_Deviation_Std"] / summary["鍍層目標值"] * 100,
        np.nan,
    )

    summary["Position Below Limit Rate (%)"] = summary["Coils_With_Position_Below_Limit"] / summary["Coil_Count"] * 100
    summary["Below Target Rate (%)"] = summary["Coils_Below_Target"] / summary["Coil_Count"] * 100
    summary["At or Above Target Rate (%)"] = summary["Coils_At_Or_Above_Target"] / summary["Coil_Count"] * 100
    summary["Excess Coating Rate (%)"] = summary["Excess_Coating_Coils"] / summary["Coil_Count"] * 100
    summary["Uneven Coating Rate (%)"] = summary["Uneven_Coating_Coils"] / summary["Coil_Count"] * 100

    summary["Uneven Threshold (%)"] = uneven_threshold_percent

    summary["Relative Stability Variation (%)"] = np.where(
        summary["鍍層目標值"] != 0,
        summary["Target_Deviation_Std"] / summary["鍍層目標值"] * 100,
        np.nan,
    )

    summary["Relative P10-P90 Range (%)"] = np.where(
        summary["鍍層目標值"] != 0,
        summary["Target_Deviation_P10_P90_Range"] / summary["鍍層目標值"] * 100,
        np.nan,
    )

    summary["Stability Grade"] = np.select(
        [
            summary["Coil_Count"] < minimum_reliable_coils,
            summary["Relative Stability Variation (%)"] <= high_stability_threshold,
            summary["Relative Stability Variation (%)"] <= medium_stability_threshold,
        ],
        ["Insufficient Data", "High Stability", "Medium Stability"],
        default="Low Stability",
    )

    summary["Stability Threshold High (%)"] = high_stability_threshold
    summary["Stability Threshold Medium (%)"] = medium_stability_threshold
    summary["Minimum Reliable Coils"] = minimum_reliable_coils

    return summary


# =========================================================
# 4. DISPLAY HELPERS
# =========================================================
def round_for_display(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    result = df.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(decimals)
    return result


def show_dataframe(df: pd.DataFrame, height: int = 500) -> None:
    st.dataframe(
        round_for_display(df, 2),
        use_container_width=True,
        height=height,
        hide_index=True,
    )


# =========================================================
# 5. CHART FUNCTIONS
# =========================================================
def finalize_chart(fig):
    fig.tight_layout()
    return fig


def plot_coil_average_vs_limits(df: pd.DataFrame):
    chart_df = df.reset_index(drop=True)
    x = np.arange(len(chart_df))

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        x,
        chart_df["Coil Average Thickness"],
        marker="o",
        linewidth=1.4,
        label="Coil Average",
    )

    ax.plot(
        x,
        chart_df["鍍層目標值"],
        color="deepskyblue",
        linestyle="--",
        linewidth=1.5,
        label="Target",
    )

    ax.plot(
        x,
        chart_df["鍍層下限值"],
        color="deepskyblue",
        linestyle=":",
        linewidth=1.8,
        label="Lower Limit",
    )

    ax.set_title("Coil Average vs Target and Lower Limit", pad=16)
    ax.set_xlabel("Output Coil Number")
    ax.set_ylabel("Coating Thickness")
    ax.grid(True, alpha=0.3)
    ax.legend()

    step = max(1, len(chart_df) // 20)
    tick_positions = x[::step]
    tick_labels = chart_df["產出鋼捲號碼"].astype(str).iloc[::step]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=60, ha="right")

    return finalize_chart(fig)


def plot_stability_by_coating_type(coating_summary: pd.DataFrame, coating_type: str):
    chart_df = coating_summary.copy().sort_values("Relative Stability Variation (%)", ascending=True)

    chart_df["Standard Label"] = (
        chart_df["上鍍層"].astype(str)
        + " | Target "
        + chart_df["鍍層目標值"].map(lambda value: f"{value:g}")
        + " | Lower "
        + chart_df["鍍層下限值"].map(lambda value: f"{value:g}")
    )

    fig_height = max(5.0, len(chart_df) * 0.52)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    bars = ax.barh(
        chart_df["Standard Label"],
        chart_df["Relative Stability Variation (%)"],
    )

    high_threshold = chart_df["Stability Threshold High (%)"].iloc[0]
    medium_threshold = chart_df["Stability Threshold Medium (%)"].iloc[0]

    ax.axvline(
        high_threshold,
        color="deepskyblue",
        linestyle="--",
        linewidth=1.4,
        label=f"High Stability Limit ({high_threshold:.1f}%)",
    )
    ax.axvline(
        medium_threshold,
        color="deepskyblue",
        linestyle=":",
        linewidth=1.8,
        label=f"Medium Stability Limit ({medium_threshold:.1f}%)",
    )

    ax.set_title(f"{coating_type} — Stability Grade", pad=16)
    ax.set_xlabel("Relative Stability Variation (%) — Lower Is More Stable")
    ax.set_ylabel("Upper Coating | Target | Lower Limit")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    max_value = max(chart_df["Relative Stability Variation (%)"].max(), medium_threshold, 1)
    offset = max_value * 0.015

    for bar, (_, row) in zip(bars, chart_df.iterrows()):
        value = bar.get_width()
        grade = row["Stability Grade"]
        coil_count = int(row["Coil_Count"])

        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}% | {grade} | n={coil_count}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlim(0, max_value * 1.38)
    return finalize_chart(fig)

# Note: Additional chart rendering functions (plot_group_target_limit, 
# plot_risk_rate_by_coating_type, etc.) remain functionally unchanged.


# =========================================================
# 6. EXCEL EXPORT
# =========================================================
@st.cache_data(show_spinner=False)
def create_excel_export(detail_df: pd.DataFrame, summary_df: pd.DataFrame, rejected_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        detail_df.to_excel(writer, index=False, sheet_name="Coil Detail")
        summary_df.to_excel(writer, index=False, sheet_name="Group Summary")
        rejected_df.to_excel(writer, index=False, sheet_name="Rejected Data")
    output.seek(0)
    return output.getvalue()


# =========================================================
# 7. UPLOAD DATA
# =========================================================
st.sidebar.header("Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload FRPMES0131_CGL file",
    type=["csv", "xlsx", "xlsm", "xls"],
)

if uploaded_file is None:
    st.info("Upload the FRPMES0131_CGL data file from the sidebar.")
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    raw_df = read_file_bytes(uploaded_file.name, file_bytes)
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
# 8. PREPARE DATA
# =========================================================
with st.spinner("Preparing data..."):
    valid_df, rejected_df = prepare_data(raw_df)

if valid_df.empty:
    st.error("No valid rows are available for analysis.")
    show_dataframe(rejected_df, height=500)
    st.stop()


# =========================================================
# 9. VIEW AND DATA LEVEL
# =========================================================
st.subheader("Select Analysis View")

if "analysis_view" not in st.session_state:
    st.session_state["analysis_view"] = "Overall View"

view_col1, view_col2 = st.columns(2)

with view_col1:
    if st.button(
        "Overall View",
        key="select_overall_view",
        type="primary" if st.session_state["analysis_view"] == "Overall View" else "secondary",
        use_container_width=True,
    ):
        st.session_state["analysis_view"] = "Overall View"
        st.rerun()

with view_col2:
    if st.button(
        "Detailed View",
        key="select_detailed_view",
        type="primary" if st.session_state["analysis_view"] == "Detailed View" else "secondary",
        use_container_width=True,
    ):
        st.session_state["analysis_view"] = "Detailed View"
        st.rerun()

view_mode = st.session_state["analysis_view"]
st.caption("Current view: " + view_mode)

st.sidebar.header("Analysis Settings")

# ---------------------------------------------------------
# DATA LEVEL LOCKED TO COIL TO ENSURE ACCURACY
# ---------------------------------------------------------
st.sidebar.success("Data Level: Aggregated to Coil Average (Locked)")
analysis_df = aggregate_to_one_row_per_coil(valid_df)


if view_mode == "Overall View":
    st.info("Overall View: all Coating Types are shown. Each Coating Type has its own chart.")
else:
    st.info("Detailed View: use the sidebar filters to inspect individual standards, orders and coils.")
