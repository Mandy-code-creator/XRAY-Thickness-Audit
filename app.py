import base64
import io
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    "Version 2026-08-04 ALLOWABLE-MARGIN RISK BUILD — "
    "North / Center / South = Top + Back"
)
st.caption(
    "Analyze coating thickness by Coating Type → Upper Coating "
    "and compare each coil with Target and Lower Limit (Based on Average Thickness)."
)


# =========================================================
# 2. CONSTANTS
# =========================================================
REQUIRED_COLUMNS = [
    "線別",
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
    "線別",
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


PRODUCTION_DATE_ALIASES = [
    "生產日期",
    "生產日期生產日期",
    "生產日",
    "Production Date",
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
def read_file_bytes(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        encodings = ["utf-8-sig", "utf-8", "big5", "cp950"]
        last_error = None
        for encoding in encodings:
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Unable to read CSV file: {last_error}")
    if lower_name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    raise ValueError("Supported formats: CSV, XLSX, XLSM and XLS.")


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def calculate_derived_metrics(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate engineering metrics after validation or coil aggregation."""
    valid = valid_df.copy()
    if valid.empty:
        return valid

    valid["North Thickness"] = valid["XRAY_A_T_N"] + valid["XRAY_A_B_N"]
    valid["Center Thickness"] = valid["XRAY_A_T_C"] + valid["XRAY_A_B_C"]
    valid["South Thickness"] = valid["XRAY_A_T_S"] + valid["XRAY_A_B_S"]
    position_columns = ["North Thickness", "Center Thickness", "South Thickness"]
    
    valid["Coil Average Thickness"] = valid[position_columns].mean(axis=1)
    valid["Target Deviation"] = valid["Coil Average Thickness"] - valid["鍍層目標值"]
    valid["Absolute Target Deviation"] = valid["Target Deviation"].abs()
    valid["Target Deviation (%)"] = np.where(
        valid["鍍層目標值"].ne(0),
        valid["Target Deviation"] / valid["鍍層目標值"] * 100,
        np.nan,
    )
    
    valid["Lower Limit Margin"] = valid["Coil Average Thickness"] - valid["鍍層下限值"]
    valid["Lower Limit Margin (%)"] = np.where(
        valid["鍍層下限值"].ne(0),
        valid["Lower Limit Margin"] / valid["鍍層下限值"] * 100,
        np.nan,
    )
    
    for position in ["North", "Center", "South"]:
        thickness_column = f"{position} Thickness"
        valid[f"{position} Target Deviation"] = valid[thickness_column] - valid["鍍層目標值"]
        valid[f"{position} Lower Limit Margin"] = valid[thickness_column] - valid["鍍層下限值"]

    valid["Minimum Position Thickness"] = valid[position_columns].min(axis=1)
    valid["Maximum Position Thickness"] = valid[position_columns].max(axis=1)
    valid["Cross-Width Range"] = valid["Maximum Position Thickness"] - valid["Minimum Position Thickness"]
    valid["Cross-Width Range (%)"] = np.where(
        valid["Coil Average Thickness"].ne(0),
        valid["Cross-Width Range"] / valid["Coil Average Thickness"] * 100,
        np.nan,
    )
    valid["Allowable Cross-Width Margin"] = valid["鍍層目標值"] - valid["鍍層下限值"]
    valid["Cross-Width Margin Utilization (%)"] = np.where(
        valid["Allowable Cross-Width Margin"].gt(0),
        valid["Cross-Width Range"] / valid["Allowable Cross-Width Margin"] * 100,
        np.nan,
    )
    
    valid["Minimum Position"] = valid[position_columns].idxmin(axis=1).map({
        "North Thickness": "North",
        "Center Thickness": "Center",
        "South Thickness": "South",
    })
    
    # -------------------------------------------------------------
    # CẬP NHẬT LOGIC: ĐÁNH GIÁ MỎNG (NG) DỰA TRÊN ĐỘ DÀY TRUNG BÌNH
    # -------------------------------------------------------------
    valid["Average Below Lower Limit"] = valid["Coil Average Thickness"] < valid["鍍層下限值"]
    valid["Average Below Target"] = valid["Coil Average Thickness"] < valid["鍍層目標值"]
    
    valid["Coil Status"] = np.select(
        [valid["Average Below Lower Limit"], valid["Average Below Target"]],
        ["Average Below Lower Limit", "Average Below Target"],
        default="Meets Requirement",
    )
    return valid


def resolve_production_date_column(data: pd.DataFrame) -> pd.DataFrame:
    """Create a standardized 生產日期 column from known source-column aliases."""
    result = data.copy()

    if "生產日期" in result.columns:
        return result

    for alias in PRODUCTION_DATE_ALIASES:
        if alias in result.columns:
            result["生產日期"] = result[alias]
            return result

    # Keep the app and report usable even when the source has no date column.
    result["生產日期"] = pd.NaT
    return result


def parse_production_date(series: pd.Series) -> pd.Series:
    """Parse normal date strings and Excel serial dates safely."""
    raw = series.copy()
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    numeric = pd.to_numeric(raw, errors="coerce")

    # Typical Excel serial dates are roughly 20,000–80,000.
    excel_mask = numeric.between(20000, 80000, inclusive="both")
    if excel_mask.any():
        parsed.loc[excel_mask] = pd.to_datetime(
            numeric.loc[excel_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    normal_mask = ~excel_mask
    if normal_mask.any():
        parsed.loc[normal_mask] = pd.to_datetime(
            raw.loc[normal_mask],
            errors="coerce",
        )

    return parsed


@st.cache_data(show_spinner=False)
def prepare_data(source_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = normalize_columns(source_df)
    data = resolve_production_date_column(data)

    for column in TEXT_COLUMNS:
        data[column] = data[column].astype("string").str.strip()

    data["生產日期"] = parse_production_date(data["生產日期"])

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column].astype(str).str.replace(",", "", regex=False).str.replace("，", "", regex=False).str.strip(),
            errors="coerce",
        )

    invalid_text = pd.DataFrame({
        column: data[column].isna() | data[column].astype("string").str.strip().eq("")
        for column in ["線別", "鍍製別", "上鍍層", "產出鋼捲號碼"]
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
    def append_reason(mask: pd.Series, text: str) -> None:
        nonlocal reasons
        current = reasons.loc[mask]
        reasons.loc[mask] = np.where(current.ne(""), current + "; " + text, text)

    for column in ["線別", "鍍製別", "上鍍層", "產出鋼捲號碼"]:
        append_reason(invalid_text[column], f"Missing {column}")
    for column in NUMERIC_COLUMNS:
        append_reason(invalid_numeric[column], f"Invalid or missing {column}")
        
    append_reason(target_invalid, "Target must be greater than 0")
    append_reason(lower_invalid, "Lower Limit must be greater than 0")
    append_reason(lower_above_target, "Lower Limit is greater than Target")

    data["Reject Reason"] = reasons
    rejected = data.loc[rejected_mask].copy()
    valid_base = data.loc[~rejected_mask].copy()
    
    return calculate_derived_metrics(valid_base), rejected


@st.cache_data(show_spinner=False)
def aggregate_to_one_row_per_coil(valid_df: pd.DataFrame) -> pd.DataFrame:
    if valid_df.empty:
        return valid_df.copy()
    group_columns = ["線別", "鍍製別", "上鍍層", "鍍層目標值", "鍍層下限值", "訂單號碼", "產出鋼捲號碼"]
    aggregation = {
        "生產日期": "min",
        "XRAY_A_T_N": "mean", "XRAY_A_T_C": "mean", "XRAY_A_T_S": "mean",
        "XRAY_A_B_N": "mean", "XRAY_A_B_C": "mean", "XRAY_A_B_S": "mean",
    }
    grouped = valid_df.groupby(group_columns, dropna=False, as_index=False).agg(aggregation)
    return calculate_derived_metrics(grouped)


@st.cache_data(show_spinner=False)
def build_group_summary(
    df: pd.DataFrame,
    cross_width_margin_limit_percent: float = 100.0,
    over_coating_threshold_percent: float = 3.0,
    high_stability_threshold: float = 3.0,
    medium_stability_threshold: float = 6.0,
    minimum_reliable_coils: int = 5,
) -> pd.DataFrame:
    """Build one summary row per coating standard."""
    if df.empty:
        return pd.DataFrame()

    # Recalculate derived columns defensively so cached/older intermediate
    # DataFrames cannot cause missing-column KeyErrors after code updates.
    working = calculate_derived_metrics(df.copy())

    if "Average Below Lower Limit" not in working.columns:
        working["Average Below Lower Limit"] = (
            working["Coil Average Thickness"] < working["鍍層下限值"]
        )

    if "Average Below Target" not in working.columns:
        working["Average Below Target"] = (
            working["Coil Average Thickness"] < working["鍍層目標值"]
        )

    working["Below Lower Limit Flag"] = (
        working["Average Below Lower Limit"].fillna(False).astype(int)
    )
    working["Below Target Flag"] = (
        working["Average Below Target"].fillna(False).astype(int)
    )
    working["At or Above Target Flag"] = (
        working["Coil Average Thickness"] >= working["鍍層目標值"]
    ).astype(int)

    # Risk 1 — Under-coating quality risk (Updated to use Average Thickness)
    working["Under-Coating Severity (%)"] = np.where(
        working["鍍層下限值"].ne(0),
        np.maximum(
            working["鍍層下限值"] - working["Coil Average Thickness"], 0
        ) / working["鍍層下限值"] * 100,
        np.nan,
    )

    # Risk 2 — Significant over-coating cost risk.
    working["Over-Coating Margin (%)"] = np.where(
        working["鍍層目標值"].ne(0),
        (working["Coil Average Thickness"] - working["鍍層目標值"])
        / working["鍍層目標值"] * 100,
        np.nan,
    )
    working["Significant Over-Coating Flag"] = (
        working["Over-Coating Margin (%)"] > over_coating_threshold_percent
    ).astype(int)

    # Risk 3 — Cross-width variation relative to the allowable engineering margin.
    working["Allowable Cross-Width Margin"] = (
        working["鍍層目標值"] - working["鍍層下限值"]
    )
    working["Cross-Width Margin Utilization (%)"] = np.where(
        working["Allowable Cross-Width Margin"].gt(0),
        working["Cross-Width Range"]
        / working["Allowable Cross-Width Margin"] * 100,
        np.nan,
    )
    working["Cross-Width Variation Flag"] = (
        working["Cross-Width Margin Utilization (%)"]
        > cross_width_margin_limit_percent
    ).astype(int)

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
            Average_Below_Lower_Limit_Coils=("Below Lower Limit Flag", "sum"),
            Coils_Below_Target=("Below Target Flag", "sum"),
            Coils_At_Or_Above_Target=("At or Above Target Flag", "sum"),
            Significant_Over_Coating_Coils=("Significant Over-Coating Flag", "sum"),
            Cross_Width_Variation_Coils=("Cross-Width Variation Flag", "sum"),
            Average_Cross_Width_Range=("Cross-Width Range", "mean"),
            Average_Cross_Width_Range_Percent=("Cross-Width Range (%)", "mean"),
            Average_Allowable_Cross_Width_Margin=("Allowable Cross-Width Margin", "mean"),
            Average_Cross_Width_Margin_Utilization=("Cross-Width Margin Utilization (%)", "mean"),
            Median_Cross_Width_Margin_Utilization=("Cross-Width Margin Utilization (%)", "median"),
        ).reset_index()
    )

    summary["Thickness_Std"] = summary["Thickness_Std"].fillna(0)
    summary["Target_Deviation_Std"] = summary["Target_Deviation_Std"].fillna(0)
    summary["Target_Deviation_P10_P90_Range"] = (
        summary["Target_Deviation_P90"] - summary["Target_Deviation_P10"]
    )
    summary["Deviation_CV_vs_Target (%)"] = np.where(
        summary["鍍層目標值"] != 0,
        summary["Target_Deviation_Std"] / summary["鍍層目標值"] * 100,
        np.nan,
    )
    summary["Average Below Limit Rate (%)"] = (
        summary["Average_Below_Lower_Limit_Coils"] / summary["Coil_Count"] * 100
    )
    summary["Below Target Rate (%)"] = (
        summary["Coils_Below_Target"] / summary["Coil_Count"] * 100
    )
    summary["At or Above Target Rate (%)"] = (
        summary["Coils_At_Or_Above_Target"] / summary["Coil_Count"] * 100
    )
    summary["Significant Over-Coating Rate (%)"] = (
        summary["Significant_Over_Coating_Coils"] / summary["Coil_Count"] * 100
    )
    summary["Cross-Width Variation Risk Rate (%)"] = (
        summary["Cross_Width_Variation_Coils"] / summary["Coil_Count"] * 100
    )
    summary["Cross-Width Margin Limit (%)"] = cross_width_margin_limit_percent
    summary["Over-Coating Threshold (%)"] = over_coating_threshold_percent

    # Priority score
    summary["Risk Priority Score"] = (
        summary["Average Below Limit Rate (%)"] * 0.50
        + summary["Cross-Width Variation Risk Rate (%)"] * 0.30
        + summary["Significant Over-Coating Rate (%)"] * 0.20
    )

    risk_columns = {
        "Under-Coating": "Average Below Limit Rate (%)",
        "Over-Coating": "Significant Over-Coating Rate (%)",
        "Cross-Width Variation": "Cross-Width Variation Risk Rate (%)",
    }
    summary["Main Risk"] = summary.apply(
        lambda row: max(risk_columns, key=lambda name: row[risk_columns[name]]),
        axis=1,
    )
    summary["Risk Priority"] = np.select(
        [
            summary["Coil_Count"] < minimum_reliable_coils,
            summary["Risk Priority Score"] >= 50,
            summary["Risk Priority Score"] >= 30,
            summary["Risk Priority Score"] >= 15,
        ],
        ["Insufficient Data", "Critical", "High", "Medium"],
        default="Low",
    )

    summary["Relative Stability Variation (%)"] = np.where(
        summary["鍍層目標值"] != 0,
        summary["Target_Deviation_Std"] / summary["鍍層目標值"] * 100,
        np.nan,
    )
    summary["Relative P10-P90 Range (%)"] = np.where(
        summary["鍍層目標值"] != 0,
        summary["Target_Deviation_P10_P90_Range"]
        / summary["鍍層目標值"] * 100,
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
# 4. DISPLAY & IMAGE HELPERS
# =========================================================
def round_for_display(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    result = df.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(decimals)
    return result

def show_dataframe(df: pd.DataFrame, height: int = 500) -> None:
    st.dataframe(round_for_display(df, 2), use_container_width=True, height=height, hide_index=True)

def fig_to_base64(fig) -> str:
    """Convert Matplotlib figure to base64 string for HTML export."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# =========================================================
# 5. CHART FUNCTIONS
# =========================================================
def finalize_chart(fig):
    fig.tight_layout()
    return fig


def plot_target_limit_by_coating_type(
    coating_summary: pd.DataFrame,
    coating_type: str,
):
    """Plot target and lower-limit differences without overlapping labels."""
    chart_df = (
        coating_summary.copy()
        .sort_values("Average_Target_Deviation", ascending=True)
        .reset_index(drop=True)
    )

    if chart_df.empty:
        fig, ax = plt.subplots(figsize=(12, 3), dpi=150)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = (
        chart_df["上鍍層"].astype(str)
        + " | Target "
        + chart_df["鍍層目標值"].map(lambda value: f"{value:g}")
        + " | Lower "
        + chart_df["鍍層下限值"].map(lambda value: f"{value:g}")
    )

    # Increase spacing between standards and between the two bars in each group.
    row_step = 1.35
    y = np.arange(len(chart_df), dtype=float) * row_step
    bar_offset = 0.26
    bar_height = 0.22

    fig_height = max(6.0, len(chart_df) * 0.92 + 1.8)
    fig, ax = plt.subplots(figsize=(15.5, fig_height), dpi=150)

    target_bars = ax.barh(
        y - bar_offset,
        chart_df["Average_Target_Deviation"],
        height=bar_height,
        label="Average Target Deviation",
        zorder=3,
    )
    lower_bars = ax.barh(
        y + bar_offset,
        chart_df["Average_Lower_Limit_Margin"],
        height=bar_height,
        label="Average Lower-Limit Margin",
        zorder=3,
    )

    ax.axvline(0, linewidth=1.2, color="#333333", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(
        f"{coating_type} — Target and Lower-Limit Difference",
        pad=18,
        fontweight="bold",
    )
    ax.set_xlabel("Thickness Difference")
    ax.grid(True, axis="x", alpha=0.25, zorder=1)
    ax.legend(loc="best", fontsize=9)

    all_values = np.concatenate(
        [
            chart_df["Average_Target_Deviation"].to_numpy(dtype=float),
            chart_df["Average_Lower_Limit_Margin"].to_numpy(dtype=float),
        ]
    )
    data_min = float(np.nanmin(all_values))
    data_max = float(np.nanmax(all_values))
    data_span = max(data_max - data_min, abs(data_min), abs(data_max), 1.0)

    # Extra room is reserved for the long count labels on both sides.
    left_padding = data_span * 0.42
    right_padding = data_span * 0.52
    ax.set_xlim(min(data_min, 0) - left_padding, max(data_max, 0) + right_padding)

    label_offset = data_span * 0.018
    label_box = {
        "boxstyle": "round,pad=0.16",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.88,
    }

    def add_bar_label(bar, value: float, text: str) -> None:
        x = value + label_offset if value >= 0 else value - label_offset
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.6,
            color="#1e293b",
            bbox=label_box,
            clip_on=False,
            zorder=5,
        )

    for bar, (_, row) in zip(target_bars, chart_df.iterrows()):
        value = float(bar.get_width())
        total = int(row["Coil_Count"])
        below_target = int(row["Coils_Below_Target"])
        at_or_above_target = total - below_target
        add_bar_label(
            bar,
            value,
            f"{value:.2f}  (≥ Target: {at_or_above_target}/{total} | < Target: {below_target})",
        )

    for bar, (_, row) in zip(lower_bars, chart_df.iterrows()):
        value = float(bar.get_width())
        total = int(row["Coil_Count"])
        below_limit = int(row["Average_Below_Lower_Limit_Coils"])
        at_or_above_limit = total - below_limit
        add_bar_label(
            bar,
            value,
            f"{value:.2f}  (≥ Lower: {at_or_above_limit}/{total} | < Lower: {below_limit})",
        )

    # Keep each standard visually separated.
    for center_y in y:
        ax.axhline(
            center_y + row_step / 2,
            color="#e2e8f0",
            linewidth=0.7,
            zorder=0,
        )

    ax.set_ylim(y[-1] + row_step * 0.65, y[0] - row_step * 0.65)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.90, bottom=0.10)
    return fig


def plot_stability_by_coating_type(coating_summary: pd.DataFrame, coating_type: str):
    chart_df = coating_summary.copy().sort_values("Relative Stability Variation (%)", ascending=True)
    chart_df["Standard Label"] = (
        chart_df["上鍍層"].astype(str)
        + " | Target " + chart_df["鍍層目標值"].map(lambda value: f"{value:g}")
        + " | Lower " + chart_df["鍍層下限值"].map(lambda value: f"{value:g}")
    )
    fig_height = max(5.0, len(chart_df) * 0.52)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=150)
    bars = ax.barh(chart_df["Standard Label"], chart_df["Relative Stability Variation (%)"])

    high_threshold = chart_df["Stability Threshold High (%)"].iloc[0]
    medium_threshold = chart_df["Stability Threshold Medium (%)"].iloc[0]

    ax.axvline(high_threshold, linestyle="--", linewidth=1.4, color="deepskyblue", label=f"High Stability Limit ({high_threshold:.1f}%)")
    ax.axvline(medium_threshold, linestyle=":", linewidth=1.8, color="deepskyblue", label=f"Medium Stability Limit ({medium_threshold:.1f}%)")

    ax.set_title(f"{coating_type} — Stability Grade", pad=16)
    ax.set_xlabel("Relative Stability Variation (%) — Lower Is More Stable")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    max_value = max(chart_df["Relative Stability Variation (%)"].max(), medium_threshold, 1)
    offset = max_value * 0.015

    for bar, (_, row) in zip(bars, chart_df.iterrows()):
        value = bar.get_width()
        grade = row["Stability Grade"]
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:.2f}% | {grade} | n={int(row['Coil_Count'])}", va="center", ha="left", fontsize=8)
    
    ax.set_xlim(0, max_value * 1.38)
    return finalize_chart(fig)


def plot_three_risks_by_coating_type(coating_summary: pd.DataFrame, coating_type: str):
    """Ranked risk heatmap with affected-coil counts and rates in every risk cell."""
    chart_df = coating_summary.copy()
    if chart_df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    chart_df["Standard Label"] = (
        chart_df["上鍍層"].astype(str)
        + " | Target " + chart_df["鍍層目標值"].map(lambda value: f"{value:g}")
        + " | Lower " + chart_df["鍍層下限值"].map(lambda value: f"{value:g}")
        + " | Total coils=" + chart_df["Coil_Count"].fillna(0).astype(int).astype(str)
    )

    risk_columns = [
        "Average Below Limit Rate (%)",
        "Significant Over-Coating Rate (%)",
        "Cross-Width Variation Risk Rate (%)",
        "Risk Priority Score",
    ]
    count_columns = [
        "Average_Below_Lower_Limit_Coils",
        "Significant_Over_Coating_Coils",
        "Cross_Width_Variation_Coils",
        None,
    ]
    column_labels = [
        "Under-Coating\nQuality Risk",
        "Over-Coating\nCost Risk",
        "Cross-Width Range\nvs Allowable Margin",
        "Priority\nScore",
    ]

    for column in risk_columns:
        chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce").fillna(0).clip(0, 100)
    for column in [c for c in count_columns if c is not None]:
        chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce").fillna(0).astype(int)
    chart_df["Coil_Count"] = pd.to_numeric(chart_df["Coil_Count"], errors="coerce").fillna(0).astype(int)

    chart_df = chart_df.sort_values(
        ["Risk Priority Score", "Average Below Limit Rate (%)", "Cross-Width Variation Risk Rate (%)"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    matrix = chart_df[risk_columns].to_numpy(dtype=float)
    row_count = len(chart_df)

    risk_cmap = LinearSegmentedColormap.from_list(
        "crisp_risk_blue",
        ["#F8FAFC", "#BAE6FD", "#0EA5E9", "#0284C7", "#082F49"],
    )

    fig_height = max(6.2, row_count * 0.72 + 2.2)
    fig, ax = plt.subplots(figsize=(14.6, fig_height), dpi=150)

    image = ax.imshow(matrix, cmap=risk_cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, fontsize=11, fontweight="bold", linespacing=1.15)
    ax.set_yticks(np.arange(row_count))
    ax.set_yticklabels(chart_df["Standard Label"], fontsize=10, fontweight="500")
    
    ax.set_title(
        f"{coating_type} — Coating Risk Heatmap",
        pad=22,
        fontsize=15,
        fontweight="bold",
    )

    for row_index in range(row_count):
        total = int(chart_df.loc[row_index, "Coil_Count"])
        for column_index in range(len(column_labels)):
            value = matrix[row_index, column_index]
            
            text_color = "white" if value >= 45 else "#000000"

            if column_index < 3:
                affected = int(chart_df.loc[row_index, count_columns[column_index]])
                cell_text = f"{affected}/{total} coils\n{value:.1f}%"
                fontweight = "bold"
            else:
                cell_text = f"{value:.1f}%"
                fontweight = "bold"

            ax.text(
                column_index,
                row_index,
                cell_text,
                ha="center",
                va="center",
                fontsize=10.5,
                linespacing=1.25,
                fontweight=fontweight,
                color=text_color,
            )
            
    ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, row_count, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(
        axis="x",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        pad=11,
        length=0,
    )
    ax.tick_params(axis="y", pad=7, length=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#697985")

    colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.035)
    colorbar.set_label("Flagged-Coil Rate / Priority Score (%)", fontsize=10.5, labelpad=10)
    colorbar.ax.tick_params(labelsize=9.5)
    colorbar.outline.set_linewidth(0.8)
    colorbar.outline.set_edgecolor("#697985")

    ax.set_xlabel(
        "Each risk cell shows flagged coils / total coils and the corresponding rate. "
        "Priority Score is a weighted index.",
        labelpad=18,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.34, right=0.93, top=0.84, bottom=0.12)
    return fig

def render_three_risk_guide() -> None:
    st.markdown(
        """
        **How to read**

        - `Flagged / Total coils` shows the actual number of coils exceeding each risk rule.
        - **Cross-Width Variation Risk** checks whether the difference between the thickest and thinnest positions is greater than the allowed margin (`Target − Lower Limit`).
        - A coil is flagged only when that utilization exceeds the selected limit (default: 100%).
        - **Priority Score** is a weighted index, not a defective-coil rate.
        - Darker cells indicate a higher flagged-coil rate; small samples are reference only.
        """
    )



# =========================================================
# 6. HTML REPORT EXPORT (FOR MANAGEMENT)
# =========================================================
@st.cache_data(show_spinner=False)
def create_html_report(summary_df: pd.DataFrame, filtered_df: pd.DataFrame) -> str:
    """Generate an HTML report using the same chart functions displayed in the app."""
    total_coils = filtered_df["產出鋼捲號碼"].nunique()
    
    if "線別" in filtered_df.columns and not filtered_df["線別"].dropna().empty:
        line_names = sorted(filtered_df["線別"].dropna().astype(str).unique().tolist())
        production_line = " / ".join(line_names)
    else:
        production_line = "未提供 (Unknown Line)"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>XRAY 鍍層厚度管理報告</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px auto; max-width: 1350px; color: #333; line-height: 1.6; }}
            h1 {{ color: #1a4f76; text-align: center; border-bottom: 2px solid #1a4f76; padding-bottom: 10px; }}
            h2 {{ color: #20639b; margin-top: 40px; border-left: 5px solid #20639b; padding-left: 10px; background-color: #f4f8fb; }}
            h3 {{ color: #333; margin-top: 30px; }}
            .scope {{ background: #f7f9fb; border: 1px solid #dfe7ee; padding: 14px 18px; border-radius: 5px; margin-bottom: 30px; line-height: 1.8; }}
            .chart-container {{ margin-bottom: 32px; page-break-inside: avoid; }}
            .chart-conclusion {{ margin: 10px 0 4px; padding: 9px 12px; border-left: 4px solid #4f7b95; background: #f4f8fb; font-size: 12px; line-height: 1.45; text-align: left; }}
            .chart-conclusion strong {{ color: #1f4e68; }}
            .risk-layout {{ display: flex; gap: 22px; align-items: flex-start; }}
            .risk-chart {{ flex: 1 1 78%; text-align: center; }}
            .risk-guide {{ flex: 0 0 250px; border: 1px solid #d9e2ea; background: #f8fafc; border-radius: 6px; padding: 14px 16px; font-size: 13px; }}
            .risk-guide h4 {{ margin-top: 0; color: #1a4f76; }}
            .risk-guide ul {{ padding-left: 18px; margin-bottom: 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            
            .table-wrap {{ width: 100%; overflow-x: auto; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-radius: 8px; background: #fff;}}
            table.custom-table {{ border-collapse: collapse; width: 100%; font-size: 13px; table-layout: fixed; }}
            table.custom-table th, table.custom-table td {{ border: 1px solid #e2e8f0; padding: 10px 8px; text-align: center; vertical-align: middle; word-wrap: break-word; }}
            table.custom-table th {{ background-color: #f8fafc; font-weight: bold; color: #334155; line-height: 1.4; }}
            table.custom-table tr:nth-child(even) {{ background-color: #fcfcfc; }}
            table.custom-table tr:hover {{ background-color: #f1f5f9; }}
            
            table.custom-table th:nth-child(1), table.custom-table td:nth-child(1) {{ width: 15%; text-align: left; }}
            table.custom-table th:nth-child(2), table.custom-table td:nth-child(2) {{ width: 5%; }}
            table.custom-table th:nth-child(3), table.custom-table td:nth-child(3),
            table.custom-table th:nth-child(4), table.custom-table td:nth-child(4),
            table.custom-table th:nth-child(5), table.custom-table td:nth-child(5) {{ width: 9%; }}
            table.custom-table th:nth-child(6), table.custom-table td:nth-child(6) {{ width: 7%; }}
            table.custom-table th:nth-child(7), table.custom-table td:nth-child(7),
            table.custom-table th:nth-child(8), table.custom-table td:nth-child(8) {{ width: 8%; }}
            table.custom-table th:nth-child(9), table.custom-table td:nth-child(9) {{ width: 30%; text-align: left; }}
            
            .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; display: inline-block; }}
            .badge-critical {{ background-color: #fee2e2; color: #991b1b; }}
            .badge-high {{ background-color: #ffedd5; color: #9a3412; }}
            .badge-medium {{ background-color: #fef3c7; color: #92400e; }}
            .badge-low {{ background-color: #f0fdf4; color: #166534; }}
            .badge-ins {{ background-color: #f1f5f9; color: #475569; }}
            @page {{ size: A4 landscape; margin: 10mm; }}
            @media print {{
                body {{ margin: 0; max-width: none; font-size: 11px; }}
                h1 {{ font-size: 18px; }}
                h2 {{ font-size: 14px; margin-top: 18px; }}
                h3 {{ font-size: 12px; margin-top: 14px; }}
                .chart-container {{ margin-bottom: 15px; page-break-inside: avoid; }}
                .chart-conclusion {{ font-size: 10px; padding: 6px 8px; margin-top: 6px; }}
                .table-wrap {{ overflow: visible; page-break-inside: auto; box-shadow: none; border: 1px solid #e2e8f0; }}
                table.custom-table {{ font-size: 9.5px; }}
                table.custom-table th, table.custom-table td {{ padding: 4px 3px; }}
                table.custom-table tr {{ page-break-inside: avoid; }}
                img {{ box-shadow: none; padding: 4px; }}
            }}
        </style>
    </head>
    <body>
        <h1>XRAY 鍍層厚度管理報告 (XRAY Coating Thickness Audit)</h1>
        <div class="scope">
            <strong>產線 (Line):</strong> {production_line} &nbsp;|&nbsp;
            <strong>分析期間 (Analysis Period):</strong> {analysis_period} <br>
            <strong>資料層級 (Analysis Level):</strong> 一捲一筆 (One row per coil) &nbsp;|&nbsp;
            <strong>總捲數 (Total Output Coils):</strong> {total_coils:,}
        </div>
    """

    coating_types = sorted(summary_df["鍍製別"].dropna().astype(str).unique().tolist())

    for c_type in coating_types:
        c_summary = summary_df[summary_df["鍍製別"].astype(str) == c_type].copy()
        html += f"<h2>鍍製別 (Coating Type): {c_type}</h2>"

        fig_target = plot_target_limit_by_coating_type(c_summary, c_type)
        html += f"""
        <div class='chart-container'>
            <h3>1. 目標與下限差異 (Target and Lower-Limit Difference)</h3>
            <img src='data:image/png;base64,{fig_to_base64(fig_target)}'>
        </div>
        """
        plt.close(fig_target)

        fig_stab = plot_stability_by_coating_type(c_summary, c_type)
        html += f"""
        <div class='chart-container'>
            <h3>2. 生產穩定度 (Stability Grade)</h3>
            <img src='data:image/png;base64,{fig_to_base64(fig_stab)}'>
        </div>
        """
        plt.close(fig_stab)

        fig_risk = plot_three_risks_by_coating_type(c_summary, c_type)
        html += f"""
        <div class='chart-container'>
            <h3>3. 鍍層風險熱圖 (Coating Risk Heatmap)</h3>
            <div class='risk-layout'>
                <div class='risk-chart'><img src='data:image/png;base64,{fig_to_base64(fig_risk)}'></div>
                <div class='risk-guide'>
                    <h4>How to read</h4>
                    <ul>
                        <li>Each risk cell shows <strong>flagged coils / total coils</strong> and the corresponding rate.</li>
                        <li><strong>Cross-Width Variation Risk</strong> checks whether the difference between the thickest and thinnest positions is greater than the allowed margin (Target − Lower Limit).</li>
                        <li>A coil is flagged only when the selected margin-utilization limit is exceeded.</li>
                        <li><strong>Priority Score</strong> is a weighted index, not a defective-coil rate.</li>
                        <li>Darker cells indicate higher risk.</li>
                        <li>Rows are ranked by Priority Score from high to low.</li>
                        <li>Small total coil counts are for reference only.</li>
                    </ul>
                </div>
            </div>
        </div>
        """
        plt.close(fig_risk)

        display_table = c_summary.copy().sort_values("Risk Priority Score", ascending=False)
        
        display_table["規格<br>(Standard)"] = display_table.apply(
            lambda row: f"<span style='font-size:13px; font-weight:600; color:#0f172a;'>{row['上鍍層']}</span><br><span style='color:#64748b; font-size:10px;'>T {row['鍍層目標值']:g} / L {row['鍍層下限值']:g}</span>",
            axis=1,
        )
        
        display_table["總捲數<br>(Total)"] = display_table["Coil_Count"].astype(int)
        
        display_table["偏薄風險<br>(Under-Coating)"] = display_table.apply(
            lambda row: f"<span style='font-size:13px; font-weight:bold; color:#e11d48;'>{row['Average Below Limit Rate (%)']:.1f}%</span><br><span style='color:#64748b; font-size:11px;'>{int(row['Average_Below_Lower_Limit_Coils'])} / {int(row['Coil_Count'])}</span>",
            axis=1,
        )
        display_table["過厚風險<br>(Over-Coating)"] = display_table.apply(
            lambda row: f"<span style='font-size:13px; font-weight:bold; color:#ea580c;'>{row['Significant Over-Coating Rate (%)']:.1f}%</span><br><span style='color:#64748b; font-size:11px;'>{int(row['Significant_Over_Coating_Coils'])} / {int(row['Coil_Count'])}</span>",
            axis=1,
        )
        display_table["橫向變異風險<br>(Cross-Width)"] = display_table.apply(
            lambda row: f"<span style='font-size:13px; font-weight:bold; color:#0284c7;'>{row['Cross-Width Variation Risk Rate (%)']:.1f}%</span><br><span style='color:#64748b; font-size:11px;'>{int(row['Cross_Width_Variation_Coils'])} / {int(row['Coil_Count'])}</span>",
            axis=1,
        )
        
        display_table["優先分數<br>(Priority Score)"] = display_table["Risk Priority Score"].map(lambda x: f"<b>{x:.1f}</b>")
        
        def get_risk_badge(risk):
            mapping = {
                "Critical": "<span class='badge badge-critical'>Critical</span>",
                "High": "<span class='badge badge-high'>High</span>",
                "Medium": "<span class='badge badge-medium'>Medium</span>",
                "Low": "<span class='badge badge-low'>Low</span>",
                "Insufficient Data": "<span class='badge badge-ins'>Insufficient</span>",
            }
            return mapping.get(risk, risk)
            
        display_table["風險等級<br>(Risk Level)"] = display_table["Risk Priority"].map(get_risk_badge)
        
        stab_map = {
            "High Stability": "高穩定",
            "Medium Stability": "中等",
            "Low Stability": "低穩定",
            "Insufficient Data": "資料不足"
        }
        display_table["生產穩定度<br>(Stability)"] = display_table["Stability Grade"].map(lambda x: stab_map.get(x, x))
        
        def generate_insight(row):
            if row["Risk Priority"] == "Insufficient Data":
                return "<span style='color:#64748b; font-size:12px;'>樣本數不足，結果僅供參考。</span>"
            if row["Risk Priority"] == "Low":
                return "<span style='color:#166534; font-size:12px; font-weight:bold;'>✓ 各項指標良好，請維持現行設定。</span>"

            insights = []
            
            under_rate = row['Average Below Limit Rate (%)']
            if under_rate >= 20:
                insights.append("<span style='color:#e11d48; font-weight:bold;'>• 高度客訴風險</span>：偏薄比例過高，強烈建議調高目標值或排查設備。")
            elif under_rate > 0:
                insights.append("<span style='color:#e11d48;'>• 具偏薄風險</span>：需微調目標值或注意風刀控制。")

            cross_rate = row['Cross-Width Variation Risk Rate (%)']
            if cross_rate >= 50:
                insights.append("<span style='color:#0284c7; font-weight:bold;'>• 橫向均勻性差</span>：耗盡容許區間，需優先檢修風刀或氣壓設定。")

            over_rate = row['Significant Over-Coating Rate (%)']
            if over_rate >= 50 and under_rate < 10:
                insights.append("<span style='color:#ea580c; font-weight:bold;'>• 嚴重過厚</span>：導致成本浪費，建議在穩定現況下調降目標值。")
            elif over_rate >= 20 and under_rate == 0:
                insights.append("<span style='color:#ea580c;'>• 具過厚情形</span>：完全無偏薄風險，有節省鋅耗之空間。")

            if not insights:
                return "<span style='color:#475569; font-size:12px;'>需持續關注異常波動。</span>"

            return f"<div style='font-size:12px; line-height:1.5;'>{'<br>'.join(insights)}</div>"

        display_table["處置建議<br>(Action / Insight)"] = display_table.apply(generate_insight, axis=1)

        final_cols = [
            "規格<br>(Standard)", "總捲數<br>(Total)", "偏薄風險<br>(Under-Coating)", 
            "過厚風險<br>(Over-Coating)", "橫向變異風險<br>(Cross-Width)", 
            "優先分數<br>(Priority Score)", "風險等級<br>(Risk Level)", 
            "生產穩定度<br>(Stability)", "處置建議<br>(Action / Insight)"
        ]
        
        table_html = display_table[final_cols].to_html(
            index=False,
            escape=False,
            classes="custom-table",
            border=0,
        )
        
        html += f"<h3>📊 數據摘要 (Data Summary)</h3><div class='table-wrap'>{table_html}</div><hr>"

    html += "</body></html>"
    return html


# =========================================================
# 7. EXCEL EXPORT
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
# 8. UPLOAD DATA
# =========================================================
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload FRPMES0131_CGL file", type=["csv", "xlsx", "xlsm", "xls"])

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
    st.stop()


# =========================================================
# 9. PREPARE DATA
# =========================================================
with st.spinner("Preparing data..."):
    valid_df, rejected_df = prepare_data(raw_df)

if valid_df.empty:
    st.error("No valid rows are available for analysis.")
    st.stop()


# =========================================================
# 10. VIEW AND DATA LEVEL
# =========================================================
st.subheader("Select Analysis View")

if "analysis_view" not in st.session_state:
    st.session_state["analysis_view"] = "Overall View"

view_col1, view_col2 = st.columns(2)

with view_col1:
    if st.button("Overall View", type="primary" if st.session_state["analysis_view"] == "Overall View" else "secondary", use_container_width=True):
        st.session_state["analysis_view"] = "Overall View"
        st.rerun()

with view_col2:
    if st.button("Detailed View", type="primary" if st.session_state["analysis_view"] == "Detailed View" else "secondary", use_container_width=True):
        st.session_state["analysis_view"] = "Detailed View"
        st.rerun()

view_mode = st.session_state["analysis_view"]
st.caption("Current view: " + view_mode)

st.sidebar.header("Analysis Settings")
st.sidebar.info("Analysis level is fixed at one row per coil to keep all tables and charts consistent.")
analysis_df = aggregate_to_one_row_per_coil(valid_df)


# =========================================================
# 11. DETAILED FILTERS
# =========================================================
if view_mode == "Detailed View":
    st.sidebar.header("Filters")
    coating_type_options = sorted(analysis_df["鍍製別"].dropna().astype(str).unique().tolist())
    selected_coating_types = st.sidebar.multiselect("Coating Type", coating_type_options, default=coating_type_options)
    filtered_df = analysis_df[analysis_df["鍍製別"].astype(str).isin(selected_coating_types)].copy()

    upper_coating_options = sorted(filtered_df["上鍍層"].dropna().astype(str).unique().tolist())
    selected_upper_coatings = st.sidebar.multiselect("Upper Coating", upper_coating_options, default=upper_coating_options)
    filtered_df = filtered_df[filtered_df["上鍍層"].astype(str).isin(selected_upper_coatings)].copy()

    order_options = sorted(filtered_df["訂單號碼"].dropna().astype(str).unique().tolist())
    selected_orders = st.sidebar.multiselect("Order Number", order_options, default=[])

    if selected_orders:
        filtered_df = filtered_df[filtered_df["訂單號碼"].astype(str).isin(selected_orders)].copy()

    coil_keyword = st.sidebar.text_input("Search Output Coil Number", value="")
    if coil_keyword.strip():
        filtered_df = filtered_df[filtered_df["產出鋼捲號碼"].astype(str).str.contains(coil_keyword.strip(), case=False, na=False)].copy()
else:
    filtered_df = analysis_df.copy()

filtered_df = filtered_df.sort_values(
    ["線別", "鍍製別", "上鍍層", "訂單號碼", "產出鋼捲號碼"]
).reset_index(drop=True)

# Recalculate all derived metrics after filtering.
# This prevents KeyError when Streamlit cache returns an older DataFrame
# created before new derived columns were added.
filtered_df = calculate_derived_metrics(filtered_df)

production_dates_ui = pd.to_datetime(
    filtered_df.get("生產日期", pd.Series(index=filtered_df.index, dtype="datetime64[ns]")),
    errors="coerce",
).dropna()
if not production_dates_ui.empty:
    ui_start = production_dates_ui.min().strftime("%Y-%m-%d")
    ui_end = production_dates_ui.max().strftime("%Y-%m-%d")
    ui_period = ui_start if ui_start == ui_end else f"{ui_start} ~ {ui_end}"
    st.caption(f"Analysis Period (生產日期): {ui_period}")


# =========================================================
# 12. KPI AND SUMMARY
# =========================================================
if view_mode == "Overall View":
    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        over_coating_threshold_percent = st.slider(
            "Significant Over-Coating Allowance (%)", 0.5, 15.0, 3.0, 0.5
        )
    with risk_col2:
        cross_width_margin_limit_percent = st.slider(
            "Cross-Width Variation Limit (% of Allowable Margin)",
            25.0,
            200.0,
            100.0,
            5.0,
            help=(
                "100% means the North/Center/South range equals Target minus Lower Limit. "
                "Only coils above this limit are flagged."
            ),
        )

    stability_col1, stability_col2, stability_col3 = st.columns(3)
    with stability_col1:
        high_stability_threshold = st.number_input("High Stability Limit (%)", 0.1, 20.0, 3.0, 0.5)
    with stability_col2:
        medium_stability_threshold = st.number_input("Medium Stability Limit (%)", 0.5, 30.0, 6.0, 0.5)
    with stability_col3:
        minimum_reliable_coils = st.number_input("Minimum Coil Count for Judgment", 2, 100, 5, 1)
else:
    over_coating_threshold_percent = 3.0
    cross_width_margin_limit_percent = 100.0
    high_stability_threshold = 3.0
    medium_stability_threshold = 6.0
    minimum_reliable_coils = 5

summary_df = build_group_summary(
    filtered_df,
    cross_width_margin_limit_percent=cross_width_margin_limit_percent,
    over_coating_threshold_percent=over_coating_threshold_percent,
    high_stability_threshold=high_stability_threshold,
    medium_stability_threshold=medium_stability_threshold,
    minimum_reliable_coils=minimum_reliable_coils,
)

coil_count = filtered_df["產出鋼捲號碼"].nunique()

below_limit_series = filtered_df.get(
    "Average Below Lower Limit",
    filtered_df["Coil Average Thickness"] < filtered_df["鍍層下限值"],
)
below_target_series = filtered_df.get(
    "Average Below Target",
    filtered_df["Coil Average Thickness"] < filtered_df["鍍層目標值"],
)

below_limit_count = int(pd.Series(below_limit_series, index=filtered_df.index).fillna(False).sum())
below_target_count = int(pd.Series(below_target_series, index=filtered_df.index).fillna(False).sum())
average_target_deviation = filtered_df["Target Deviation"].mean()
average_lower_margin = filtered_df["Lower Limit Margin"].mean()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Coil Count", f"{coil_count:,}")
kpi2.metric("Average Below Limit", f"{below_limit_count:,}", f"{below_limit_count / len(filtered_df) * 100:.1f}%", delta_color="inverse")
kpi3.metric("Average Below Target", f"{below_target_count:,}", f"{below_target_count / len(filtered_df) * 100:.1f}%", delta_color="inverse")
kpi4.metric("Average Target Deviation", f"{average_target_deviation:.2f}")
kpi5.metric("Average Lower-Limit Margin", f"{average_lower_margin:.2f}")


# =========================================================
# 13. UI DISPLAY RENDER
# =========================================================
if view_mode == "Overall View":
    st.header("Overall View")
    overall_section = st.radio(
        "Overall Section",
        ["Target and Lower-Limit Difference", "Stability Analysis", "Risk Heatmap Analysis"],
        horizontal=True,
    )
    coating_types = sorted(summary_df["鍍製別"].dropna().astype(str).unique().tolist())

    if overall_section == "Target and Lower-Limit Difference":
        for coating_type in coating_types:
            c_summary = summary_df[summary_df["鍍製別"].astype(str) == coating_type].copy()
            st.subheader(f"Coating Type: {coating_type}")
            fig = plot_target_limit_by_coating_type(c_summary, coating_type)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.divider()

    elif overall_section == "Stability Analysis":
        for coating_type in coating_types:
            c_summary = summary_df[summary_df["鍍製別"].astype(str) == coating_type].copy()
            st.subheader(f"Coating Type: {coating_type}")
            fig = plot_stability_by_coating_type(c_summary, coating_type)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.divider()

    elif overall_section == "Risk Heatmap Analysis":
        for coating_type in coating_types:
            c_summary = summary_df[summary_df["鍍製別"].astype(str) == coating_type].copy()
            st.subheader(f"Coating Type: {coating_type}")
            chart_col, guide_col = st.columns([4.3, 1.2], vertical_alignment="top")
            with chart_col:
                fig = plot_three_risks_by_coating_type(c_summary, coating_type)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with guide_col:
                render_three_risk_guide()
            st.divider()

else:
    st.header("Detailed View")
    detailed_section = st.radio("Detailed Section", ["Management Summary", "Coil Detail"], horizontal=True)

    if detailed_section == "Management Summary":
        show_dataframe(summary_df, height=520)
    elif detailed_section == "Coil Detail":
        show_dataframe(filtered_df, height=650)


# =========================================================
# 14. REPORT EXPORTS (SIDEBAR)
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("Management Reports")
st.sidebar.info("The HTML report uses the same chart functions and current thresholds shown in the app.")

if st.sidebar.button("Prepare HTML Report", use_container_width=True):
    try:
        with st.spinner("Rendering HTML Report..."):
            html_data = create_html_report(summary_df, filtered_df)
            st.session_state["html_report"] = html_data
        st.sidebar.success("HTML report is ready.")
    except Exception as exc:
        st.sidebar.error(f"HTML report generation failed: {exc}")

if "html_report" in st.session_state:
    st.sidebar.download_button(
        label="Download HTML Report",
        data=st.session_state["html_report"],
        file_name=f"Management_Report_Boss_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
        mime="text/html",
        use_container_width=True,
        type="primary"
    )

st.sidebar.divider()
st.sidebar.subheader("Data Exports")
if st.sidebar.button("Prepare Excel Export", use_container_width=True):
    detail_cols = ["生產日期", "線別", "鍍製別", "上鍍層", "訂單號碼", "產出鋼捲號碼", "鍍層目標值", "鍍層下限值", "Coil Average Thickness", "Coil Status"]
    export_bytes = create_excel_export(filtered_df[detail_cols], summary_df, rejected_df)
    st.sidebar.download_button(
        label="Download Excel Data",
        data=export_bytes,
        file_name="XRAY_Coating_Thickness_Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
