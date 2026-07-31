
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


@st.cache_data(show_spinner=False)
def prepare_data(
    source_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculation:
        North = XRAY_A_T_N + XRAY_A_B_N
        Center = XRAY_A_T_C + XRAY_A_B_C
        South = XRAY_A_T_S + XRAY_A_B_S

        Coil Average = (North + Center + South) / 3
    """
    data = normalize_columns(source_df)

    for column in TEXT_COLUMNS:
        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
        )

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("，", "", regex=False)
            .str.strip(),
            errors="coerce",
        )

    # -----------------------------------------------------
    # Vectorized validation
    # -----------------------------------------------------
    invalid_text = pd.DataFrame(
        {
            column: (
                data[column].isna()
                | data[column].astype("string").str.strip().eq("")
            )
            for column in ["鍍製別", "上鍍層", "產出鋼捲號碼"]
        }
    )

    invalid_numeric = data[NUMERIC_COLUMNS].isna()

    target_invalid = data["鍍層目標值"].le(0)
    lower_invalid = data["鍍層下限值"].le(0)
    lower_above_target = (
        data["鍍層下限值"] > data["鍍層目標值"]
    )

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
            reasons.where(~mask, "")
            + np.where(
                reasons.ne("") & mask,
                "; ",
                "",
            )
            + f"Missing {column}",
        )

    for column in NUMERIC_COLUMNS:
        mask = invalid_numeric[column]
        reasons = reasons.mask(
            mask,
            reasons.where(~mask, "")
            + np.where(
                reasons.ne("") & mask,
                "; ",
                "",
            )
            + f"Invalid or missing {column}",
        )

    for mask, text in [
        (target_invalid, "Target must be greater than 0"),
        (lower_invalid, "Lower Limit must be greater than 0"),
        (
            lower_above_target,
            "Lower Limit is greater than Target",
        ),
    ]:
        reasons = reasons.mask(
            mask,
            reasons.where(~mask, "")
            + np.where(
                reasons.ne("") & mask,
                "; ",
                "",
            )
            + text,
        )

    data["Reject Reason"] = reasons

    rejected = data.loc[rejected_mask].copy()
    valid = data.loc[~rejected_mask].copy()

    if valid.empty:
        return valid, rejected

    # -----------------------------------------------------
    # Correct thickness calculation
    # -----------------------------------------------------
    valid["North Thickness"] = (
        valid["XRAY_A_T_N"]
        + valid["XRAY_A_B_N"]
    )

    valid["Center Thickness"] = (
        valid["XRAY_A_T_C"]
        + valid["XRAY_A_B_C"]
    )

    valid["South Thickness"] = (
        valid["XRAY_A_T_S"]
        + valid["XRAY_A_B_S"]
    )

    position_columns = [
        "North Thickness",
        "Center Thickness",
        "South Thickness",
    ]

    valid["Coil Average Thickness"] = (
        valid[position_columns].mean(axis=1)
    )

    valid["Target Deviation"] = (
        valid["Coil Average Thickness"]
        - valid["鍍層目標值"]
    )

    valid["Absolute Target Deviation"] = (
        valid["Target Deviation"].abs()
    )

    valid["Target Deviation (%)"] = (
        valid["Target Deviation"]
        / valid["鍍層目標值"]
        * 100
    )

    valid["Lower Limit Margin"] = (
        valid["Coil Average Thickness"]
        - valid["鍍層下限值"]
    )

    valid["Lower Limit Margin (%)"] = (
        valid["Lower Limit Margin"]
        / valid["鍍層下限值"]
        * 100
    )

    for position in ["North", "Center", "South"]:
        thickness_column = f"{position} Thickness"

        valid[f"{position} Target Deviation"] = (
            valid[thickness_column]
            - valid["鍍層目標值"]
        )

        valid[f"{position} Lower Limit Margin"] = (
            valid[thickness_column]
            - valid["鍍層下限值"]
        )

    valid["Minimum Position Thickness"] = (
        valid[position_columns].min(axis=1)
    )

    valid["Maximum Position Thickness"] = (
        valid[position_columns].max(axis=1)
    )

    valid["Cross-Width Range"] = (
        valid["Maximum Position Thickness"]
        - valid["Minimum Position Thickness"]
    )

    valid["Cross-Width Range (%)"] = (
        valid["Cross-Width Range"]
        / valid["Coil Average Thickness"]
        * 100
    )

    valid["Minimum Position"] = (
        valid[position_columns]
        .idxmin(axis=1)
        .map(
            {
                "North Thickness": "North",
                "Center Thickness": "Center",
                "South Thickness": "South",
            }
        )
    )

    valid["Any Position Below Lower Limit"] = (
        valid["Minimum Position Thickness"]
        < valid["鍍層下限值"]
    )

    valid["Average Below Target"] = (
        valid["Coil Average Thickness"]
        < valid["鍍層目標值"]
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

    recalculated, _ = prepare_data(grouped)
    return recalculated


@st.cache_data(show_spinner=False)
def build_group_summary(
    df: pd.DataFrame,
    uneven_threshold_percent: float = 10.0,
    high_stability_threshold: float = 3.0,
    medium_stability_threshold: float = 6.0,
    minimum_reliable_coils: int = 5,
) -> pd.DataFrame:
    """
    Grouping level:
        Coating Type -> Upper Coating -> Target -> Lower Limit

    Stability:
        standard deviation and P10-P90 range of Target Deviation

    Three risks:
        1. Any position below Lower Limit
        2. Coil average above Target (excess coating)
        3. Cross-width range percentage above selected threshold
    """
    if df.empty:
        return pd.DataFrame()

    working = df.copy()

    working["Below Lower Limit Flag"] = (
        working["Any Position Below Lower Limit"].astype(int)
    )

    working["Below Target Flag"] = (
        working["Average Below Target"].astype(int)
    )

    working["At or Above Target Flag"] = (
        working["Coil Average Thickness"]
        >= working["鍍層目標值"]
    ).astype(int)

    working["Excess Coating Flag"] = (
        working["Coil Average Thickness"]
        > working["鍍層目標值"]
    ).astype(int)

    working["Uneven Coating Flag"] = (
        working["Cross-Width Range (%)"]
        > uneven_threshold_percent
    ).astype(int)

    group_columns = [
        "鍍製別",
        "上鍍層",
        "鍍層目標值",
        "鍍層下限值",
    ]

    summary = (
        working.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            Coil_Count=("產出鋼捲號碼", "nunique"),
            Average_Thickness=(
                "Coil Average Thickness",
                "mean",
            ),
            Thickness_Std=(
                "Coil Average Thickness",
                "std",
            ),
            Average_Target_Deviation=(
                "Target Deviation",
                "mean",
            ),
            Median_Target_Deviation=(
                "Target Deviation",
                "median",
            ),
            Target_Deviation_Std=(
                "Target Deviation",
                "std",
            ),
            Target_Deviation_P10=(
                "Target Deviation",
                lambda series: series.quantile(0.10),
            ),
            Target_Deviation_P90=(
                "Target Deviation",
                lambda series: series.quantile(0.90),
            ),
            Average_Lower_Limit_Margin=(
                "Lower Limit Margin",
                "mean",
            ),
            Minimum_Lower_Limit_Margin=(
                "Lower Limit Margin",
                "min",
            ),
            Coils_With_Position_Below_Limit=(
                "Below Lower Limit Flag",
                "sum",
            ),
            Coils_Below_Target=(
                "Below Target Flag",
                "sum",
            ),
            Coils_At_Or_Above_Target=(
                "At or Above Target Flag",
                "sum",
            ),
            Excess_Coating_Coils=(
                "Excess Coating Flag",
                "sum",
            ),
            Uneven_Coating_Coils=(
                "Uneven Coating Flag",
                "sum",
            ),
            Average_Cross_Width_Range=(
                "Cross-Width Range",
                "mean",
            ),
            Average_Cross_Width_Range_Percent=(
                "Cross-Width Range (%)",
                "mean",
            ),
        )
        .reset_index()
    )

    # A group containing only one coil has no sample standard deviation.
    summary["Thickness_Std"] = summary["Thickness_Std"].fillna(0)
    summary["Target_Deviation_Std"] = (
        summary["Target_Deviation_Std"].fillna(0)
    )

    summary["Target_Deviation_P10_P90_Range"] = (
        summary["Target_Deviation_P90"]
        - summary["Target_Deviation_P10"]
    )

    summary["Deviation_CV_vs_Target (%)"] = np.where(
        summary["鍍層目標值"] != 0,
        summary["Target_Deviation_Std"]
        / summary["鍍層目標值"]
        * 100,
        np.nan,
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

    summary["Excess Coating Rate (%)"] = (
        summary["Excess_Coating_Coils"]
        / summary["Coil_Count"]
        * 100
    )

    summary["Uneven Coating Rate (%)"] = (
        summary["Uneven_Coating_Coils"]
        / summary["Coil_Count"]
        * 100
    )

    summary["Uneven Threshold (%)"] = uneven_threshold_percent

    summary["Relative Stability Variation (%)"] = np.where(
        summary["鍍層目標值"] != 0,
        summary["Target_Deviation_Std"]
        / summary["鍍層目標值"]
        * 100,
        np.nan,
    )

    summary["Relative P10-P90 Range (%)"] = np.where(
        summary["鍍層目標值"] != 0,
        summary["Target_Deviation_P10_P90_Range"]
        / summary["鍍層目標值"]
        * 100,
        np.nan,
    )

    summary["Stability Grade"] = np.select(
        [
            summary["Coil_Count"] < minimum_reliable_coils,
            summary["Relative Stability Variation (%)"]
            <= high_stability_threshold,
            summary["Relative Stability Variation (%)"]
            <= medium_stability_threshold,
        ],
        [
            "Insufficient Data",
            "High Stability",
            "Medium Stability",
        ],
        default="Low Stability",
    )

    summary["Stability Threshold High (%)"] = (
        high_stability_threshold
    )
    summary["Stability Threshold Medium (%)"] = (
        medium_stability_threshold
    )
    summary["Minimum Reliable Coils"] = (
        minimum_reliable_coils
    )

    return summary


# =========================================================
# 4. DISPLAY HELPERS
# =========================================================
def round_for_display(
    df: pd.DataFrame,
    decimals: int = 2,
) -> pd.DataFrame:
    result = df.copy()
    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns
    result[numeric_columns] = result[
        numeric_columns
    ].round(decimals)
    return result


def show_dataframe(
    df: pd.DataFrame,
    height: int = 500,
) -> None:
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


def plot_group_target_limit(
    summary_df: pd.DataFrame,
):
    chart_df = summary_df.copy()

    chart_df["Group"] = (
        chart_df["鍍製別"].astype(str)
        + " → "
        + chart_df["上鍍層"].astype(str)
    )

    chart_df["Abs Target Deviation"] = (
        chart_df["Average_Target_Deviation"].abs()
    )

    chart_df = chart_df.sort_values(
        "Abs Target Deviation",
        ascending=True,
    )

    y = np.arange(len(chart_df))
    bar_height = 0.36
    fig_height = max(6, len(chart_df) * 0.46)

    fig, ax = plt.subplots(
        figsize=(14, fig_height)
    )

    target_bars = ax.barh(
        y - bar_height / 2,
        chart_df["Average_Target_Deviation"],
        height=bar_height,
        label="Average Target Deviation",
    )

    lower_bars = ax.barh(
        y + bar_height / 2,
        chart_df["Average_Lower_Limit_Margin"],
        height=bar_height,
        label="Average Lower-Limit Margin",
    )

    ax.axvline(0, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(chart_df["Group"])
    ax.set_title(
        "Target and Lower-Limit Difference by Group",
        pad=16,
    )
    ax.set_xlabel("Thickness Difference")
    ax.set_ylabel("Coating Type → Upper Coating")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    max_abs = max(
        chart_df["Average_Target_Deviation"]
        .abs()
        .max(),
        chart_df["Average_Lower_Limit_Margin"]
        .abs()
        .max(),
        1,
    )
    offset = max_abs * 0.015

    for bars in [target_bars, lower_bars]:
        for bar in bars:
            value = bar.get_width()

            ax.text(
                value + offset if value >= 0 else value - offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )

    return finalize_chart(fig)



def plot_target_limit_by_coating_type(
    coating_summary: pd.DataFrame,
    coating_type: str,
):
    """
    One chart for one Coating Type.
    Each bar pair represents an Upper Coating.
    """
    chart_df = coating_summary.copy().sort_values(
        "Average_Target_Deviation",
        ascending=True,
    )

    labels = (
        chart_df["上鍍層"].astype(str)
        + " | Target "
        + chart_df["鍍層目標值"].map(lambda value: f"{value:g}")
        + " | Lower "
        + chart_df["鍍層下限值"].map(lambda value: f"{value:g}")
    )
    y = np.arange(len(chart_df))
    bar_height = 0.36

    fig_height = max(4.8, len(chart_df) * 0.48)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    target_bars = ax.barh(
        y - bar_height / 2,
        chart_df["Average_Target_Deviation"],
        height=bar_height,
        label="Average Target Deviation",
    )

    lower_bars = ax.barh(
        y + bar_height / 2,
        chart_df["Average_Lower_Limit_Margin"],
        height=bar_height,
        label="Average Lower-Limit Margin",
    )

    ax.axvline(0, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_title(
        f"{coating_type} — Target and Lower-Limit Difference",
        pad=16,
    )
    ax.set_xlabel("Thickness Difference")
    ax.set_ylabel("Upper Coating | Target | Lower Limit")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    max_abs = max(
        chart_df["Average_Target_Deviation"].abs().max(),
        chart_df["Average_Lower_Limit_Margin"].abs().max(),
        1,
    )
    offset = max_abs * 0.015

    for bars in [target_bars, lower_bars]:
        for bar in bars:
            value = bar.get_width()
            ax.text(
                value + offset if value >= 0 else value - offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )

    return finalize_chart(fig)


def plot_risk_rate_by_coating_type(
    coating_summary: pd.DataFrame,
    coating_type: str,
):
    """
    One risk-rate chart for one Coating Type.
    Each bar pair represents an Upper Coating.
    """
    chart_df = coating_summary.copy().sort_values(
        "Position Below Limit Rate (%)",
        ascending=True,
    )

    labels = (
        chart_df["上鍍層"].astype(str)
        + " | Target "
        + chart_df["鍍層目標值"].map(lambda value: f"{value:g}")
        + " | Lower "
        + chart_df["鍍層下限值"].map(lambda value: f"{value:g}")
    )
    y = np.arange(len(chart_df))
    bar_height = 0.36

    fig_height = max(4.8, len(chart_df) * 0.48)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    target_bars = ax.barh(
        y - bar_height / 2,
        chart_df["Below Target Rate (%)"],
        height=bar_height,
        label="Average Below Target Rate",
    )

    limit_bars = ax.barh(
        y + bar_height / 2,
        chart_df["Position Below Limit Rate (%)"],
        height=bar_height,
        label="Any Position Below Lower Limit Rate",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_title(
        f"{coating_type} — Quality Risk Rate",
        pad=16,
    )
    ax.set_xlabel("Coil Rate (%)")
    ax.set_ylabel("Upper Coating | Target | Lower Limit")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    max_rate = max(
        chart_df["Below Target Rate (%)"].max(),
        chart_df["Position Below Limit Rate (%)"].max(),
        1,
    )
    offset = max_rate * 0.012

    for bars in [target_bars, limit_bars]:
        for bar in bars:
            value = bar.get_width()
            ax.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                ha="left",
                fontsize=8,
            )

    ax.set_xlim(0, max(100, max_rate * 1.12))
    return finalize_chart(fig)


def plot_coil_count_by_coating_type(
    coating_summary: pd.DataFrame,
    coating_type: str,
):
    """One coil-count chart for one Coating Type."""
    chart_df = coating_summary.copy().sort_values(
        "Coil_Count",
        ascending=True,
    )

    fig_height = max(4.8, len(chart_df) * 0.42)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    chart_df["Standard Label"] = (
        chart_df["上鍍層"].astype(str)
        + " | Target "
        + chart_df["鍍層目標值"].map(lambda value: f"{value:g}")
        + " | Lower "
        + chart_df["鍍層下限值"].map(lambda value: f"{value:g}")
    )

    bars = ax.barh(
        chart_df["Standard Label"],
        chart_df["Coil_Count"],
    )

    ax.set_title(
        f"{coating_type} — Coil Count by Upper Coating",
        pad=16,
    )
    ax.set_xlabel("Coil Count")
    ax.set_ylabel("Upper Coating | Target | Lower Limit")
    ax.grid(True, axis="x", alpha=0.3)

    max_count = max(chart_df["Coil_Count"].max(), 1)
    offset = max_count * 0.012

    for bar, value in zip(bars, chart_df["Coil_Count"]):
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlim(0, max_count * 1.12)
    return finalize_chart(fig)



def plot_stability_by_coating_type(
    coating_summary: pd.DataFrame,
    coating_type: str,
):
    """
    Main stability chart:
    Relative Stability Variation (%) = Std(Target Deviation) / Target × 100

    Lower values mean greater stability.
    """
    chart_df = coating_summary.copy().sort_values(
        "Relative Stability Variation (%)",
        ascending=True,
    )

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

    high_threshold = (
        chart_df["Stability Threshold High (%)"].iloc[0]
    )
    medium_threshold = (
        chart_df["Stability Threshold Medium (%)"].iloc[0]
    )

    ax.axvline(
        high_threshold,
        linestyle="--",
        linewidth=1.4,
        label=f"High Stability Limit ({high_threshold:.1f}%)",
    )
    ax.axvline(
        medium_threshold,
        linestyle=":",
        linewidth=1.8,
        label=f"Medium Stability Limit ({medium_threshold:.1f}%)",
    )

    ax.set_title(
        f"{coating_type} — Stability Grade",
        pad=16,
    )
    ax.set_xlabel(
        "Relative Stability Variation (%) — Lower Is More Stable"
    )
    ax.set_ylabel("Upper Coating | Target | Lower Limit")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    max_value = max(
        chart_df["Relative Stability Variation (%)"].max(),
        medium_threshold,
        1,
    )
    offset = max_value * 0.015

    for bar, (_, row) in zip(
        bars,
        chart_df.iterrows(),
    ):
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

    ax.set_xlim(
        0,
        max_value * 1.38,
    )

    return finalize_chart(fig)


def plot_three_risks_by_coating_type(
    coating_summary: pd.DataFrame,
    coating_type: str,
):
    """
    Three non-exclusive risk rates:
    - Below Lower Limit
    - Excess Coating
    - Cross-Width Unevenness
    """
    chart_df = coating_summary.copy()

    chart_df["Standard Label"] = (
        chart_df["上鍍層"].astype(str)
        + " | Target "
        + chart_df["鍍層目標值"].map(lambda value: f"{value:g}")
        + " | Lower "
        + chart_df["鍍層下限值"].map(lambda value: f"{value:g}")
    )

    chart_df = chart_df.sort_values(
        "Position Below Limit Rate (%)",
        ascending=True,
    )

    y = np.arange(len(chart_df))
    bar_height = 0.24
    fig_height = max(5.2, len(chart_df) * 0.58)

    fig, ax = plt.subplots(figsize=(13, fig_height))

    below_bars = ax.barh(
        y - bar_height,
        chart_df["Position Below Limit Rate (%)"],
        height=bar_height,
        label="Below Lower Limit",
    )

    excess_bars = ax.barh(
        y,
        chart_df["Excess Coating Rate (%)"],
        height=bar_height,
        label="Excess Coating",
    )

    uneven_bars = ax.barh(
        y + bar_height,
        chart_df["Uneven Coating Rate (%)"],
        height=bar_height,
        label="Cross-Width Unevenness",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(chart_df["Standard Label"])
    ax.set_title(
        f"{coating_type} — Three-Risk Overview",
        pad=16,
    )
    ax.set_xlabel("Coil Risk Rate (%)")
    ax.set_ylabel("Upper Coating | Target | Lower Limit")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    max_rate = max(
        chart_df["Position Below Limit Rate (%)"].max(),
        chart_df["Excess Coating Rate (%)"].max(),
        chart_df["Uneven Coating Rate (%)"].max(),
        1,
    )
    offset = max_rate * 0.010

    for bars in [below_bars, excess_bars, uneven_bars]:
        for bar in bars:
            value = bar.get_width()
            ax.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                ha="left",
                fontsize=7.5,
            )

    ax.set_xlim(0, max(100, max_rate * 1.14))
    return finalize_chart(fig)


def plot_group_risk_rate(
    summary_df: pd.DataFrame,
):
    chart_df = summary_df.copy()

    chart_df["Group"] = (
        chart_df["鍍製別"].astype(str)
        + " → "
        + chart_df["上鍍層"].astype(str)
    )

    chart_df = chart_df.sort_values(
        "Position Below Limit Rate (%)",
        ascending=True,
    )

    y = np.arange(len(chart_df))
    bar_height = 0.36
    fig_height = max(6, len(chart_df) * 0.46)

    fig, ax = plt.subplots(
        figsize=(14, fig_height)
    )

    target_bars = ax.barh(
        y - bar_height / 2,
        chart_df["Below Target Rate (%)"],
        height=bar_height,
        label="Average Below Target Rate",
    )

    limit_bars = ax.barh(
        y + bar_height / 2,
        chart_df["Position Below Limit Rate (%)"],
        height=bar_height,
        label="Any Position Below Lower Limit Rate",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(chart_df["Group"])
    ax.set_title(
        "Quality Risk Rate by Group",
        pad=16,
    )
    ax.set_xlabel("Coil Rate (%)")
    ax.set_ylabel("Coating Type → Upper Coating")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    max_rate = max(
        chart_df["Below Target Rate (%)"].max(),
        chart_df["Position Below Limit Rate (%)"].max(),
        1,
    )
    offset = max_rate * 0.012

    for bars in [target_bars, limit_bars]:
        for bar in bars:
            value = bar.get_width()
            ax.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                ha="left",
                fontsize=8,
            )

    ax.set_xlim(
        0,
        max(100, max_rate * 1.12),
    )

    return finalize_chart(fig)


def plot_group_coil_count(
    summary_df: pd.DataFrame,
):
    chart_df = summary_df.copy()

    chart_df["Group"] = (
        chart_df["鍍製別"].astype(str)
        + " → "
        + chart_df["上鍍層"].astype(str)
    )

    chart_df = chart_df.sort_values(
        "Coil_Count",
        ascending=True,
    )

    fig_height = max(6, len(chart_df) * 0.40)
    fig, ax = plt.subplots(
        figsize=(14, fig_height)
    )

    bars = ax.barh(
        chart_df["Group"],
        chart_df["Coil_Count"],
    )

    ax.set_title(
        "Coil Count by Group",
        pad=16,
    )
    ax.set_xlabel("Coil Count")
    ax.set_ylabel("Coating Type → Upper Coating")
    ax.grid(True, axis="x", alpha=0.3)

    max_count = max(
        chart_df["Coil_Count"].max(),
        1,
    )
    offset = max_count * 0.012

    for bar, value in zip(
        bars,
        chart_df["Coil_Count"],
    ):
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlim(
        0,
        max_count * 1.12,
    )

    return finalize_chart(fig)


def plot_coil_average_vs_limits(
    df: pd.DataFrame,
):
    chart_df = df.reset_index(drop=True)
    x = np.arange(len(chart_df))

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

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

    ax.set_title(
        "Coil Average vs Target and Lower Limit",
        pad=16,
    )
    ax.set_xlabel("Output Coil Number")
    ax.set_ylabel("Coating Thickness")
    ax.grid(True, alpha=0.3)
    ax.legend()

    step = max(1, len(chart_df) // 20)
    tick_positions = x[::step]
    tick_labels = (
        chart_df["產出鋼捲號碼"]
        .astype(str)
        .iloc[::step]
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        rotation=60,
        ha="right",
    )

    return finalize_chart(fig)


def select_top_deviation_data(
    df: pd.DataFrame,
    ranking_method: str,
    top_n: int,
):
    if ranking_method == "Absolute Target Deviation":
        ranked = df.nlargest(
            top_n,
            "Absolute Target Deviation",
        )
        value_column = "Target Deviation"
        title = (
            f"Top {top_n} Coils by "
            "Absolute Target Deviation"
        )

    elif ranking_method == "Above-Target Deviation":
        ranked = df.nlargest(
            top_n,
            "Target Deviation",
        )
        value_column = "Target Deviation"
        title = f"Top {top_n} Coils Above Target"

    elif ranking_method == "Below-Target Deviation":
        ranked = df.nsmallest(
            top_n,
            "Target Deviation",
        )
        value_column = "Target Deviation"
        title = f"Top {top_n} Coils Below Target"

    elif ranking_method == "Lower-Limit Risk":
        ranked = df.nsmallest(
            top_n,
            "Lower Limit Margin",
        )
        value_column = "Lower Limit Margin"
        title = (
            f"Top {top_n} Coils Closest to "
            "or Below Lower Limit"
        )

    else:
        ranked = df.nlargest(
            top_n,
            "Cross-Width Range",
        )
        value_column = "Cross-Width Range"
        title = (
            f"Top {top_n} Coils by "
            "Cross-Width Variation"
        )

    return ranked.copy(), value_column, title


def plot_top_deviation(
    ranked_df: pd.DataFrame,
    value_column: str,
    title: str,
):
    chart_df = ranked_df.sort_values(
        value_column,
        ascending=True,
    )

    labels = (
        chart_df["產出鋼捲號碼"].astype(str)
        + " | "
        + chart_df["鍍製別"].astype(str)
        + " | "
        + chart_df["上鍍層"].astype(str)
    )

    fig_height = max(
        6,
        len(chart_df) * 0.38,
    )

    fig, ax = plt.subplots(
        figsize=(13, fig_height)
    )

    bars = ax.barh(
        labels,
        chart_df[value_column],
    )

    ax.axvline(0, linewidth=1.0)
    ax.set_title(title, pad=16)
    ax.set_xlabel(value_column)
    ax.set_ylabel(
        "Output Coil | Coating Type | Upper Coating"
    )
    ax.grid(True, axis="x", alpha=0.3)

    max_abs = max(
        chart_df[value_column].abs().max(),
        1,
    )
    offset = max_abs * 0.012

    for bar, value in zip(
        bars,
        chart_df[value_column],
    ):
        ax.text(
            value + offset if value >= 0 else value - offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )

    return finalize_chart(fig)


# =========================================================
# 6. EXCEL EXPORT
# =========================================================
@st.cache_data(show_spinner=False)
def create_excel_export(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        detail_df.to_excel(
            writer,
            index=False,
            sheet_name="Coil Detail",
        )
        summary_df.to_excel(
            writer,
            index=False,
            sheet_name="Group Summary",
        )
        rejected_df.to_excel(
            writer,
            index=False,
            sheet_name="Rejected Data",
        )

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
    st.info(
        "Upload the FRPMES0131_CGL data file "
        "from the sidebar."
    )
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    raw_df = read_file_bytes(
        uploaded_file.name,
        file_bytes,
    )
except Exception as exc:
    st.error(f"File reading failed: {exc}")
    st.stop()

raw_df = normalize_columns(raw_df)

missing_columns = validate_required_columns(raw_df)

if missing_columns:
    st.error(
        "The uploaded file is missing required columns:"
    )
    st.code("\n".join(missing_columns))

    with st.expander(
        "Columns found in the uploaded file"
    ):
        st.write(list(raw_df.columns))

    st.stop()


# =========================================================
# 8. PREPARE DATA
# =========================================================
with st.spinner("Preparing data..."):
    valid_df, rejected_df = prepare_data(raw_df)

if valid_df.empty:
    st.error(
        "No valid rows are available for analysis."
    )
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
        type=(
            "primary"
            if st.session_state["analysis_view"] == "Overall View"
            else "secondary"
        ),
        use_container_width=True,
    ):
        st.session_state["analysis_view"] = "Overall View"
        st.rerun()

with view_col2:
    if st.button(
        "Detailed View",
        key="select_detailed_view",
        type=(
            "primary"
            if st.session_state["analysis_view"] == "Detailed View"
            else "secondary"
        ),
        use_container_width=True,
    ):
        st.session_state["analysis_view"] = "Detailed View"
        st.rerun()

view_mode = st.session_state["analysis_view"]

st.caption(
    "Current view: " + view_mode
)

st.sidebar.header("Analysis Settings")

data_level = st.sidebar.radio(
    "Data Level",
    [
        "One Row per Coil",
        "Original Rows",
    ],
    index=0,
)

if view_mode == "Overall View":
    st.info(
        "Overall View: all Coating Types are shown. "
        "Each Coating Type has its own chart."
    )
else:
    st.info(
        "Detailed View: use the sidebar filters to inspect "
        "individual standards, orders and coils."
    )

if data_level == "One Row per Coil":
    analysis_df = aggregate_to_one_row_per_coil(
        valid_df
    )
else:
    analysis_df = valid_df.copy()


# =========================================================
# 10. DETAILED FILTERS
# =========================================================
if view_mode == "Detailed View":
    st.sidebar.header("Filters")

    coating_type_options = sorted(
        analysis_df["鍍製別"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_coating_types = st.sidebar.multiselect(
        "Coating Type",
        coating_type_options,
        default=coating_type_options,
    )

    filtered_df = analysis_df[
        analysis_df["鍍製別"]
        .astype(str)
        .isin(selected_coating_types)
    ].copy()

    upper_coating_options = sorted(
        filtered_df["上鍍層"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_upper_coatings = st.sidebar.multiselect(
        "Upper Coating",
        upper_coating_options,
        default=upper_coating_options,
    )

    filtered_df = filtered_df[
        filtered_df["上鍍層"]
        .astype(str)
        .isin(selected_upper_coatings)
    ].copy()

    order_options = sorted(
        filtered_df["訂單號碼"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_orders = st.sidebar.multiselect(
        "Order Number",
        order_options,
        default=[],
        help="Leave empty to include all orders.",
    )

    if selected_orders:
        filtered_df = filtered_df[
            filtered_df["訂單號碼"]
            .astype(str)
            .isin(selected_orders)
        ].copy()

    coil_keyword = st.sidebar.text_input(
        "Search Output Coil Number",
        value="",
    )

    if coil_keyword.strip():
        filtered_df = filtered_df[
            filtered_df["產出鋼捲號碼"]
            .astype(str)
            .str.contains(
                coil_keyword.strip(),
                case=False,
                na=False,
            )
        ].copy()

else:
    filtered_df = analysis_df.copy()

filtered_df = filtered_df.sort_values(
    [
        "鍍製別",
        "上鍍層",
        "訂單號碼",
        "產出鋼捲號碼",
    ]
).reset_index(drop=True)

if filtered_df.empty:
    st.warning(
        "No data matches the selected filters."
    )
    st.stop()


# =========================================================
# 11. KPI AND SUMMARY
# =========================================================
if view_mode == "Overall View":
    threshold_col1, threshold_col2, threshold_col3 = st.columns(3)

    with threshold_col1:
        uneven_threshold_percent = st.slider(
            "Cross-Width Unevenness Threshold (%)",
            min_value=1.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            help=(
                "A coil is classified as uneven when "
                "Cross-Width Range (%) exceeds this threshold."
            ),
        )

    with threshold_col2:
        high_stability_threshold = st.number_input(
            "High Stability Limit (%)",
            min_value=0.1,
            max_value=20.0,
            value=3.0,
            step=0.5,
            help=(
                "Relative Stability Variation at or below this value "
                "is classified as High Stability."
            ),
        )

    with threshold_col3:
        medium_stability_threshold = st.number_input(
            "Medium Stability Limit (%)",
            min_value=0.5,
            max_value=30.0,
            value=6.0,
            step=0.5,
            help=(
                "Values above the High limit and at or below this value "
                "are classified as Medium Stability."
            ),
        )

    minimum_reliable_coils = st.number_input(
        "Minimum Coil Count for Stability Judgment",
        min_value=2,
        max_value=100,
        value=5,
        step=1,
        help=(
            "Groups below this coil count are classified as "
            "Insufficient Data."
        ),
    )
else:
    uneven_threshold_percent = 10.0
    high_stability_threshold = 3.0
    medium_stability_threshold = 6.0
    minimum_reliable_coils = 5

summary_df = build_group_summary(
    filtered_df,
    uneven_threshold_percent,
    high_stability_threshold,
    medium_stability_threshold,
    minimum_reliable_coils,
)

coil_count = filtered_df[
    "產出鋼捲號碼"
].nunique()

below_limit_count = int(
    filtered_df[
        "Any Position Below Lower Limit"
    ].sum()
)

below_target_count = int(
    filtered_df[
        "Average Below Target"
    ].sum()
)

average_target_deviation = (
    filtered_df["Target Deviation"].mean()
)

average_lower_margin = (
    filtered_df["Lower Limit Margin"].mean()
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    "Coil Count",
    f"{coil_count:,}",
)

kpi2.metric(
    "Any Position Below Limit",
    f"{below_limit_count:,}",
    (
        f"{below_limit_count / len(filtered_df) * 100:.1f}%"
    ),
    delta_color="inverse",
)

kpi3.metric(
    "Average Below Target",
    f"{below_target_count:,}",
    (
        f"{below_target_count / len(filtered_df) * 100:.1f}%"
    ),
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
# 12. OVERALL VIEW
# =========================================================
if view_mode == "Overall View":
    st.header("Overall View")
    st.caption(
        "Each Coating Type is displayed in a separate chart. "
        "The same Upper Coating is split into separate standards whenever "
        "Target or Lower Limit is different."
    )

    overall_section = st.radio(
        "Overall Section",
        [
            "Target and Lower-Limit Difference",
            "Stability Analysis",
            "Three-Risk Analysis",
            "Quality Risk Rate",
            "Coil Count",
            "Summary Table",
        ],
        horizontal=True,
    )

    coating_types = sorted(
        summary_df["鍍製別"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if overall_section == (
        "Target and Lower-Limit Difference"
    ):
        for coating_type in coating_types:
            coating_summary = summary_df[
                summary_df["鍍製別"].astype(str) == coating_type
            ].copy()

            st.subheader(f"Coating Type: {coating_type}")

            fig = plot_target_limit_by_coating_type(
                coating_summary,
                coating_type,
            )
            st.pyplot(
                fig,
                use_container_width=True,
            )
            plt.close(fig)

            st.divider()

    elif overall_section == "Stability Analysis":
        st.info(
            "Stability is judged by Relative Stability Variation (%) = "
            "Target Deviation Std / Target × 100. "
            f"High Stability ≤ {high_stability_threshold:.1f}%; "
            f"Medium Stability ≤ {medium_stability_threshold:.1f}%; "
            "higher values are Low Stability. "
            f"Groups with fewer than {minimum_reliable_coils} coils are "
            "classified as Insufficient Data."
        )

        for coating_type in coating_types:
            coating_summary = summary_df[
                summary_df["鍍製別"].astype(str) == coating_type
            ].copy()

            st.subheader(f"Coating Type: {coating_type}")

            fig = plot_stability_by_coating_type(
                coating_summary,
                coating_type,
            )
            st.pyplot(
                fig,
                use_container_width=True,
            )
            plt.close(fig)

            stability_columns = [
                "上鍍層",
                "鍍層目標值",
                "鍍層下限值",
                "Coil_Count",
                "Average_Target_Deviation",
                "Median_Target_Deviation",
                "Target_Deviation_Std",
                "Relative Stability Variation (%)",
                "Target_Deviation_P10",
                "Target_Deviation_P90",
                "Target_Deviation_P10_P90_Range",
                "Relative P10-P90 Range (%)",
                "Stability Grade",
            ]

            show_dataframe(
                coating_summary[stability_columns],
                height=min(
                    420,
                    80 + len(coating_summary) * 36,
                ),
            )
            st.divider()

    elif overall_section == "Three-Risk Analysis":
        st.info(
            "The three risk rates are non-exclusive: one coil may be "
            "both below the Lower Limit and uneven. "
            f"Unevenness threshold: {uneven_threshold_percent:.1f}%."
        )

        for coating_type in coating_types:
            coating_summary = summary_df[
                summary_df["鍍製別"].astype(str) == coating_type
            ].copy()

            st.subheader(f"Coating Type: {coating_type}")

            fig = plot_three_risks_by_coating_type(
                coating_summary,
                coating_type,
            )
            st.pyplot(
                fig,
                use_container_width=True,
            )
            plt.close(fig)

            risk_columns = [
                "上鍍層",
                "鍍層目標值",
                "鍍層下限值",
                "Coil_Count",
                "Coils_With_Position_Below_Limit",
                "Position Below Limit Rate (%)",
                "Excess_Coating_Coils",
                "Excess Coating Rate (%)",
                "Uneven_Coating_Coils",
                "Uneven Coating Rate (%)",
                "Uneven Threshold (%)",
            ]

            show_dataframe(
                coating_summary[risk_columns],
                height=min(
                    420,
                    80 + len(coating_summary) * 36,
                ),
            )
            st.divider()

    elif overall_section == "Quality Risk Rate":
        for coating_type in coating_types:
            coating_summary = summary_df[
                summary_df["鍍製別"].astype(str) == coating_type
            ].copy()

            st.subheader(f"Coating Type: {coating_type}")

            fig = plot_risk_rate_by_coating_type(
                coating_summary,
                coating_type,
            )
            st.pyplot(
                fig,
                use_container_width=True,
            )
            plt.close(fig)

            st.divider()

    elif overall_section == "Coil Count":
        for coating_type in coating_types:
            coating_summary = summary_df[
                summary_df["鍍製別"].astype(str) == coating_type
            ].copy()

            st.subheader(f"Coating Type: {coating_type}")

            fig = plot_coil_count_by_coating_type(
                coating_summary,
                coating_type,
            )
            st.pyplot(
                fig,
                use_container_width=True,
            )
            plt.close(fig)

            st.divider()

    else:
        overall_columns = [
            "鍍製別",
            "上鍍層",
            "鍍層目標值",
            "鍍層下限值",
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
            "Target_Deviation_Std",
            "Relative Stability Variation (%)",
            "Target_Deviation_P10",
            "Target_Deviation_P90",
            "Target_Deviation_P10_P90_Range",
            "Relative P10-P90 Range (%)",
            "Stability Grade",
            "Average_Cross_Width_Range",
            "Average_Cross_Width_Range_Percent",
            "Excess_Coating_Coils",
            "Excess Coating Rate (%)",
            "Uneven_Coating_Coils",
            "Uneven Coating Rate (%)",
            "Uneven Threshold (%)",
        ]

        show_dataframe(
            summary_df[overall_columns],
            height=650,
        )


# =========================================================
# 13. DETAILED VIEW
# =========================================================
else:
    st.header("Detailed View")
    st.caption(
        "Use sidebar filters to inspect individual "
        "groups, orders and coils."
    )

    detailed_section = st.radio(
        "Detailed Section",
        [
            "Management Summary",
            "Coil Detail",
            "Charts",
            "Data Quality",
            "Export",
        ],
        horizontal=True,
    )

    if detailed_section == "Management Summary":
        summary_columns = [
            "鍍製別",
            "上鍍層",
            "鍍層目標值",
            "鍍層下限值",
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

        show_dataframe(
            summary_df[summary_columns],
            height=520,
        )

        summary_chart = st.selectbox(
            "Summary Chart",
            [
                "Target and Lower-Limit Difference",
                "Quality Risk Rate",
                "Coil Count",
            ],
        )

        if summary_chart == (
            "Target and Lower-Limit Difference"
        ):
            fig = plot_group_target_limit(summary_df)

        elif summary_chart == "Quality Risk Rate":
            fig = plot_group_risk_rate(summary_df)

        else:
            fig = plot_group_coil_count(summary_df)

        st.pyplot(
            fig,
            use_container_width=True,
        )
        plt.close(fig)

    elif detailed_section == "Coil Detail":
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

        show_dataframe(
            filtered_df[detail_columns],
            height=650,
        )

        abnormal_df = filtered_df[
            filtered_df["Coil Status"]
            != "Meets Requirement"
        ]

        st.subheader("Abnormal Coil List")

        if abnormal_df.empty:
            st.success(
                "No abnormal coils were found."
            )
        else:
            show_dataframe(
                abnormal_df[detail_columns],
                height=420,
            )

    elif detailed_section == "Charts":
        chart_type = st.selectbox(
            "Chart Type",
            [
                "Coil Trend",
                "Top Deviation Ranking",
            ],
        )

        if chart_type == "Coil Trend":
            max_rows = min(
                300,
                len(filtered_df),
            )

            rows_to_show = st.slider(
                "Number of coils shown",
                min_value=1,
                max_value=max(1, max_rows),
                value=min(50, max(1, max_rows)),
            )

            chart_df = filtered_df.head(
                rows_to_show
            )

            fig = plot_coil_average_vs_limits(
                chart_df
            )

            st.pyplot(
                fig,
                use_container_width=True,
            )
            plt.close(fig)

        else:
            ranking_method = st.selectbox(
                "Ranking Method",
                [
                    "Absolute Target Deviation",
                    "Above-Target Deviation",
                    "Below-Target Deviation",
                    "Lower-Limit Risk",
                    "Cross-Width Variation",
                ],
            )

            top_n_max = max(
                1,
                min(50, len(filtered_df)),
            )

            top_n = st.slider(
                "Top N",
                min_value=1,
                max_value=top_n_max,
                value=min(20, top_n_max),
            )

            ranked_df, value_column, title = (
                select_top_deviation_data(
                    filtered_df,
                    ranking_method,
                    top_n,
                )
            )

            fig = plot_top_deviation(
                ranked_df,
                value_column,
                title,
            )

            st.pyplot(
                fig,
                use_container_width=True,
            )
            plt.close(fig)

            ranking_columns = [
                "鍍製別",
                "上鍍層",
                "訂單號碼",
                "產出鋼捲號碼",
                "鍍層目標值",
                "鍍層下限值",
                "Coil Average Thickness",
                "Target Deviation",
                "Lower Limit Margin",
                "Cross-Width Range",
                "Coil Status",
            ]

            show_dataframe(
                ranked_df[ranking_columns],
                height=500,
            )

    elif detailed_section == "Data Quality":
        q1, q2, q3 = st.columns(3)

        q1.metric(
            "Original Rows",
            f"{len(raw_df):,}",
        )
        q2.metric(
            "Valid Rows",
            f"{len(valid_df):,}",
        )
        q3.metric(
            "Rejected Rows",
            f"{len(rejected_df):,}",
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

            show_dataframe(
                rejected_df[rejected_columns],
                height=620,
            )

    else:
        st.subheader("Export Analysis")

        st.info(
            "The Excel file is created only when "
            "you press the button below."
        )

        if st.button(
            "Prepare Excel Export",
            use_container_width=True,
        ):
            detail_columns = [
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
                "Target Deviation (%)",
                "Lower Limit Margin",
                "Lower Limit Margin (%)",
                "Minimum Position",
                "Minimum Position Thickness",
                "Cross-Width Range",
                "Cross-Width Range (%)",
                "Coil Status",
            ]

            export_bytes = create_excel_export(
                filtered_df[detail_columns],
                summary_df,
                rejected_df,
            )

            st.download_button(
                "Download Analysis Excel",
                data=export_bytes,
                file_name=(
                    "XRAY_Coating_Thickness_Analysis.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )


# =========================================================
# 14. FOOTNOTE
# =========================================================
st.caption(
    "Calculation: "
    "North = XRAY_A_T_N + XRAY_A_B_N; "
    "Center = XRAY_A_T_C + XRAY_A_B_C; "
    "South = XRAY_A_T_S + XRAY_A_B_S; "
    "Coil Average = (North + Center + South) / 3."
)
