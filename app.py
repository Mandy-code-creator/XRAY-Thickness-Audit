
import io
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="XRAY Coating Thickness Audit",
    page_icon="📏",
    layout="wide",
)

st.title("XRAY Coating Thickness Audit")
st.caption(
    "依鍍製別及上鍍層分類，分析各鋼捲北／中／南位置之實際鍍層值，"
    "並與鍍層目標值及鍍層下限值比較。"
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
# 3. HELPER FUNCTIONS
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and remove invisible spaces."""
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u3000", " ", regex=False)
        .str.strip()
    )
    return result


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel uploaded through Streamlit."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        raw = uploaded_file.getvalue()

        encodings = ["utf-8-sig", "utf-8", "big5", "cp950"]
        last_error = None

        for encoding in encodings:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=encoding)
            except Exception as exc:
                last_error = exc

        raise ValueError(f"CSV 讀取失敗：{last_error}")

    if filename.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("僅支援 CSV、XLSX、XLSM 或 XLS 檔案。")


def read_local_file(file_path: str) -> pd.DataFrame:
    """Read a local file path when the app runs on the same computer."""
    file_path = file_path.strip().strip('"')
    if not file_path:
        raise ValueError("請輸入檔案路徑。")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案：{file_path}")

    lower_path = file_path.lower()

    if lower_path.endswith(".csv"):
        encodings = ["utf-8-sig", "utf-8", "big5", "cp950"]
        last_error = None

        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except Exception as exc:
                last_error = exc

        raise ValueError(f"CSV 讀取失敗：{last_error}")

    if lower_path.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(file_path)

    raise ValueError("僅支援 CSV、XLSX、XLSM 或 XLS 檔案。")


def convert_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to numeric."""
    result = df.copy()

    for column in columns:
        result[column] = (
            result[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("，", "", regex=False)
            .str.strip()
        )
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def validate_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    return len(missing) == 0, missing


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and calculate all coating thickness metrics.

    Position value:
        North = (Top_N + Back_N) / 2
        Center = (Top_C + Back_C) / 2
        South = (Top_S + Back_S) / 2

    Coil average:
        (North + Center + South) / 3
    """
    result = normalize_columns(df)
    result = convert_numeric(result, NUMERIC_COLUMNS)

    for column in ["鍍製別", "上鍍層", "訂單號碼", "產出鋼捲號碼"]:
        result[column] = result[column].astype("string").str.strip()

    # Keep a reason for rejected rows
    rejection_reasons = []

    for _, row in result.iterrows():
        reasons = []

        if pd.isna(row["鍍製別"]) or str(row["鍍製別"]).strip() == "":
            reasons.append("缺少鍍製別")
        if pd.isna(row["上鍍層"]) or str(row["上鍍層"]).strip() == "":
            reasons.append("缺少上鍍層")
        if pd.isna(row["產出鋼捲號碼"]) or str(row["產出鋼捲號碼"]).strip() == "":
            reasons.append("缺少產出鋼捲號碼")

        for column in NUMERIC_COLUMNS:
            if pd.isna(row[column]):
                reasons.append(f"{column}非數值或空白")

        if pd.notna(row["鍍層目標值"]) and row["鍍層目標值"] <= 0:
            reasons.append("鍍層目標值需大於0")

        if pd.notna(row["鍍層下限值"]) and row["鍍層下限值"] <= 0:
            reasons.append("鍍層下限值需大於0")

        if (
            pd.notna(row["鍍層目標值"])
            and pd.notna(row["鍍層下限值"])
            and row["鍍層下限值"] > row["鍍層目標值"]
        ):
            reasons.append("鍍層下限值高於目標值")

        rejection_reasons.append("；".join(reasons))

    result["Reject_Reason"] = rejection_reasons

    rejected = result[result["Reject_Reason"] != ""].copy()
    valid = result[result["Reject_Reason"] == ""].copy()

    # Calculate position values
    valid["北側鍍層值"] = (
        valid["XRAY_A_T_N"] + valid["XRAY_A_B_N"]
    ) / 2.0

    valid["中央鍍層值"] = (
        valid["XRAY_A_T_C"] + valid["XRAY_A_B_C"]
    ) / 2.0

    valid["南側鍍層值"] = (
        valid["XRAY_A_T_S"] + valid["XRAY_A_B_S"]
    ) / 2.0

    # Coil average
    valid["鋼捲平均鍍層值"] = valid[
        ["北側鍍層值", "中央鍍層值", "南側鍍層值"]
    ].mean(axis=1)

    # Difference from target
    valid["目標差異"] = (
        valid["鋼捲平均鍍層值"] - valid["鍍層目標值"]
    )

    valid["目標差異率(%)"] = np.where(
        valid["鍍層目標值"] != 0,
        valid["目標差異"] / valid["鍍層目標值"] * 100,
        np.nan,
    )

    # Margin above lower limit
    valid["下限餘裕"] = (
        valid["鋼捲平均鍍層值"] - valid["鍍層下限值"]
    )

    valid["下限餘裕率(%)"] = np.where(
        valid["鍍層下限值"] != 0,
        valid["下限餘裕"] / valid["鍍層下限值"] * 100,
        np.nan,
    )

    # Position-level difference from target and lower limit
    for label, column in [
        ("北側", "北側鍍層值"),
        ("中央", "中央鍍層值"),
        ("南側", "南側鍍層值"),
    ]:
        valid[f"{label}目標差異"] = valid[column] - valid["鍍層目標值"]
        valid[f"{label}下限餘裕"] = valid[column] - valid["鍍層下限值"]

    # Horizontal uniformity
    position_cols = ["北側鍍層值", "中央鍍層值", "南側鍍層值"]

    valid["最低位置鍍層值"] = valid[position_cols].min(axis=1)
    valid["最高位置鍍層值"] = valid[position_cols].max(axis=1)
    valid["橫向最大差異"] = (
        valid["最高位置鍍層值"] - valid["最低位置鍍層值"]
    )

    valid["橫向差異率(%)"] = np.where(
        valid["鋼捲平均鍍層值"] != 0,
        valid["橫向最大差異"] / valid["鋼捲平均鍍層值"] * 100,
        np.nan,
    )

    valid["最低位置"] = valid[position_cols].idxmin(axis=1).map(
        {
            "北側鍍層值": "北側",
            "中央鍍層值": "中央",
            "南側鍍層值": "南側",
        }
    )

    # Coil judgments
    valid["任一位置低於下限"] = (
        valid["最低位置鍍層值"] < valid["鍍層下限值"]
    )

    valid["平均低於下限"] = (
        valid["鋼捲平均鍍層值"] < valid["鍍層下限值"]
    )

    valid["平均低於目標"] = (
        valid["鋼捲平均鍍層值"] < valid["鍍層目標值"]
    )

    valid["鋼捲判定"] = np.select(
        [
            valid["任一位置低於下限"],
            valid["平均低於目標"],
        ],
        [
            "任一位置低於下限",
            "平均低於目標",
        ],
        default="符合要求",
    )

    return valid, rejected


def aggregate_by_coil(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the same coil appears multiple times, aggregate to one row per coil.
    Numeric measurements use mean; target/lower limit use median.
    """
    if df.empty:
        return df.copy()

    group_cols = ["鍍製別", "上鍍層", "訂單號碼", "產出鋼捲號碼"]

    agg_map = {
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
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(agg_map)
    )

    grouped, _ = prepare_data(grouped)
    return grouped


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create management summary by 鍍製別 → 上鍍層."""
    if df.empty:
        return pd.DataFrame()

    working = df.copy()

    working["低於下限鋼捲"] = working["任一位置低於下限"].astype(int)
    working["平均低於目標鋼捲"] = working["平均低於目標"].astype(int)
    working["高於或等於目標鋼捲"] = (
        working["鋼捲平均鍍層值"] >= working["鍍層目標值"]
    ).astype(int)

    summary = (
        working.groupby(["鍍製別", "上鍍層"], dropna=False)
        .agg(
            鋼捲數=("產出鋼捲號碼", "nunique"),
            平均鍍層值=("鋼捲平均鍍層值", "mean"),
            目標差異平均=("目標差異", "mean"),
            目標差異中位數=("目標差異", "median"),
            下限餘裕平均=("下限餘裕", "mean"),
            最小下限餘裕=("下限餘裕", "min"),
            任一位置低於下限鋼捲數=("低於下限鋼捲", "sum"),
            平均低於目標鋼捲數=("平均低於目標鋼捲", "sum"),
            高於或等於目標鋼捲數=("高於或等於目標鋼捲", "sum"),
            平均橫向最大差異=("橫向最大差異", "mean"),
            平均橫向差異率=("橫向差異率(%)", "mean"),
        )
        .reset_index()
    )

    summary["任一位置低於下限率(%)"] = np.where(
        summary["鋼捲數"] != 0,
        summary["任一位置低於下限鋼捲數"] / summary["鋼捲數"] * 100,
        np.nan,
    )

    summary["平均低於目標率(%)"] = np.where(
        summary["鋼捲數"] != 0,
        summary["平均低於目標鋼捲數"] / summary["鋼捲數"] * 100,
        np.nan,
    )

    summary["高於或等於目標率(%)"] = np.where(
        summary["鋼捲數"] != 0,
        summary["高於或等於目標鋼捲數"] / summary["鋼捲數"] * 100,
        np.nan,
    )

    return summary


def dataframe_to_excel_bytes(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        detail_df.to_excel(writer, index=False, sheet_name="Coil_Detail")
        summary_df.to_excel(writer, index=False, sheet_name="Group_Summary")
        rejected_df.to_excel(writer, index=False, sheet_name="Rejected_Data")

    output.seek(0)
    return output.getvalue()


def format_table(df: pd.DataFrame, decimals: int = 2):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df.style.format({col: f"{{:.{decimals}f}}" for col in numeric_cols})


def plot_coil_actual_vs_limits(df: pd.DataFrame):
    plot_df = df.reset_index(drop=True).copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(plot_df))

    ax.plot(
        x,
        plot_df["鋼捲平均鍍層值"],
        marker="o",
        linewidth=1.5,
        label="Coil Average",
    )
    ax.plot(
        x,
        plot_df["鍍層目標值"],
        linestyle="--",
        linewidth=1.5,
        label="Target",
    )
    ax.plot(
        x,
        plot_df["鍍層下限值"],
        linestyle=":",
        linewidth=1.8,
        label="Lower Limit",
    )

    ax.set_title("Coil Average vs Target and Lower Limit", pad=16)
    ax.set_xlabel("Output Coil Number")
    ax.set_ylabel("Coating Thickness")
    ax.grid(True, alpha=0.3)
    ax.legend()

    step = max(1, len(plot_df) // 20)
    tick_index = x[::step]
    tick_labels = plot_df["產出鋼捲號碼"].astype(str).iloc[::step]

    ax.set_xticks(tick_index)
    ax.set_xticklabels(tick_labels, rotation=60, ha="right")

    for spine in ax.spines.values():
        spine.set_visible(True)

    fig.tight_layout()
    return fig


def plot_target_deviation(df: pd.DataFrame):
    plot_df = df.reset_index(drop=True).copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(plot_df))

    ax.bar(x, plot_df["目標差異"])
    ax.axhline(0, linewidth=1.2)

    ax.set_title("Target Deviation by Coil", pad=16)
    ax.set_xlabel("Output Coil Number")
    ax.set_ylabel("Actual Average - Target")
    ax.grid(True, axis="y", alpha=0.3)

    step = max(1, len(plot_df) // 20)
    tick_index = x[::step]
    tick_labels = plot_df["產出鋼捲號碼"].astype(str).iloc[::step]

    ax.set_xticks(tick_index)
    ax.set_xticklabels(tick_labels, rotation=60, ha="right")

    for spine in ax.spines.values():
        spine.set_visible(True)

    fig.tight_layout()
    return fig


def plot_position_profile(df: pd.DataFrame):
    plot_df = df.reset_index(drop=True).copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(plot_df))

    ax.plot(x, plot_df["北側鍍層值"], marker="o", label="North")
    ax.plot(x, plot_df["中央鍍層值"], marker="o", label="Center")
    ax.plot(x, plot_df["南側鍍層值"], marker="o", label="South")
    ax.plot(
        x,
        plot_df["鍍層下限值"],
        linestyle=":",
        linewidth=1.8,
        label="Lower Limit",
    )

    ax.set_title("North / Center / South Coating Profile", pad=16)
    ax.set_xlabel("Output Coil Number")
    ax.set_ylabel("Coating Thickness")
    ax.grid(True, alpha=0.3)
    ax.legend()

    step = max(1, len(plot_df) // 20)
    tick_index = x[::step]
    tick_labels = plot_df["產出鋼捲號碼"].astype(str).iloc[::step]

    ax.set_xticks(tick_index)
    ax.set_xticklabels(tick_labels, rotation=60, ha="right")

    for spine in ax.spines.values():
        spine.set_visible(True)

    fig.tight_layout()
    return fig


# =========================================================
# 4. DATA SOURCE
# =========================================================
st.sidebar.header("Data Source")

source_mode = st.sidebar.radio(
    "Choose data source",
    ["Upload file", "Local file path"],
)

raw_df = None

if source_mode == "Upload file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload FRPMES0131_CGL file",
        type=["csv", "xlsx", "xlsm", "xls"],
    )

    if uploaded_file is not None:
        try:
            raw_df = read_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"檔案讀取失敗：{exc}")
            st.stop()

else:
    default_path = r"D:\Mandy\Source data\鍍鋅\鍍一\XRAY Thickness Audit\FRPMES0131_CGL.xlsx"

    local_path = st.sidebar.text_input(
        "Local file path",
        value=default_path,
    )

    load_button = st.sidebar.button("Load local file", use_container_width=True)

    if load_button:
        try:
            raw_df = read_local_file(local_path)
            st.session_state["local_raw_df"] = raw_df
        except Exception as exc:
            st.error(f"本機檔案讀取失敗：{exc}")
            st.stop()

    if "local_raw_df" in st.session_state:
        raw_df = st.session_state["local_raw_df"]


if raw_df is None:
    st.info("請先上傳資料檔，或輸入本機檔案完整路徑。")
    st.stop()


# =========================================================
# 5. VALIDATION AND PREPARATION
# =========================================================
raw_df = normalize_columns(raw_df)

is_valid, missing_columns = validate_columns(raw_df)

if not is_valid:
    st.error("資料缺少必要欄位：")
    st.code("\n".join(missing_columns))
    st.write("目前欄位：")
    st.write(list(raw_df.columns))
    st.stop()

valid_df, rejected_df = prepare_data(raw_df)

if valid_df.empty:
    st.error("沒有可分析的有效資料。請檢查欄位內容與數值格式。")
    if not rejected_df.empty:
        st.dataframe(rejected_df, use_container_width=True)
    st.stop()


# =========================================================
# 6. ANALYSIS LEVEL
# =========================================================
st.sidebar.header("Analysis Settings")

analysis_level = st.sidebar.radio(
    "Data level",
    ["One row per coil", "Original rows"],
    help=(
        "若同一產出鋼捲號碼出現多筆資料，One row per coil 會先彙整為一筆。"
    ),
)

if analysis_level == "One row per coil":
    analysis_df = aggregate_by_coil(valid_df)
else:
    analysis_df = valid_df.copy()


# =========================================================
# 7. FILTERS
# =========================================================
st.sidebar.header("Filters")

coating_types = sorted(
    analysis_df["鍍製別"].dropna().astype(str).unique().tolist()
)

selected_coating_types = st.sidebar.multiselect(
    "鍍製別",
    options=coating_types,
    default=coating_types,
)

filtered_df = analysis_df[
    analysis_df["鍍製別"].astype(str).isin(selected_coating_types)
].copy()

upper_coatings = sorted(
    filtered_df["上鍍層"].dropna().astype(str).unique().tolist()
)

selected_upper_coatings = st.sidebar.multiselect(
    "上鍍層",
    options=upper_coatings,
    default=upper_coatings,
)

filtered_df = filtered_df[
    filtered_df["上鍍層"].astype(str).isin(selected_upper_coatings)
].copy()

orders = sorted(
    filtered_df["訂單號碼"].dropna().astype(str).unique().tolist()
)

selected_orders = st.sidebar.multiselect(
    "訂單號碼",
    options=orders,
    default=[],
    help="未選擇時顯示全部訂單。",
)

if selected_orders:
    filtered_df = filtered_df[
        filtered_df["訂單號碼"].astype(str).isin(selected_orders)
    ].copy()

coil_search = st.sidebar.text_input(
    "產出鋼捲號碼搜尋",
    value="",
)

if coil_search.strip():
    filtered_df = filtered_df[
        filtered_df["產出鋼捲號碼"]
        .astype(str)
        .str.contains(coil_search.strip(), case=False, na=False)
    ].copy()

filtered_df = filtered_df.sort_values(
    ["鍍製別", "上鍍層", "訂單號碼", "產出鋼捲號碼"]
).reset_index(drop=True)


# =========================================================
# 8. MAIN DASHBOARD
# =========================================================
if filtered_df.empty:
    st.warning("目前篩選條件沒有資料。")
    st.stop()

summary_df = build_summary(filtered_df)

total_coils = filtered_df["產出鋼捲號碼"].nunique()
below_lower_count = int(filtered_df["任一位置低於下限"].sum())
below_target_count = int(filtered_df["平均低於目標"].sum())
avg_target_diff = filtered_df["目標差異"].mean()
avg_lower_margin = filtered_df["下限餘裕"].mean()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("鋼捲數", f"{total_coils:,}")
kpi2.metric(
    "任一位置低於下限",
    f"{below_lower_count:,}",
    f"{below_lower_count / len(filtered_df) * 100:.1f}%",
    delta_color="inverse",
)
kpi3.metric(
    "平均低於目標",
    f"{below_target_count:,}",
    f"{below_target_count / len(filtered_df) * 100:.1f}%",
    delta_color="inverse",
)
kpi4.metric("平均目標差異", f"{avg_target_diff:.2f}")
kpi5.metric("平均下限餘裕", f"{avg_lower_margin:.2f}")


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Management Summary",
        "Coil Detail",
        "Charts",
        "Data Quality",
    ]
)


# =========================================================
# TAB 1. MANAGEMENT SUMMARY
# =========================================================
with tab1:
    st.subheader("Summary by 鍍製別 → 上鍍層")

    summary_display_cols = [
        "鍍製別",
        "上鍍層",
        "鋼捲數",
        "平均鍍層值",
        "目標差異平均",
        "目標差異中位數",
        "下限餘裕平均",
        "最小下限餘裕",
        "任一位置低於下限鋼捲數",
        "任一位置低於下限率(%)",
        "平均低於目標鋼捲數",
        "平均低於目標率(%)",
        "高於或等於目標鋼捲數",
        "高於或等於目標率(%)",
        "平均橫向最大差異",
        "平均橫向差異率",
    ]

    st.dataframe(
        format_table(summary_df[summary_display_cols], 2),
        use_container_width=True,
        height=460,
    )

    st.markdown(
        """
        **判讀重點**

        - `目標差異`：鋼捲平均鍍層值 − 鍍層目標值。
        - `下限餘裕`：鋼捲平均鍍層值 − 鍍層下限值。
        - `任一位置低於下限率`：北、中央、南任一位置低於下限之鋼捲比例。
        - `橫向最大差異`：北、中央、南三位置最大值 − 最小值。
        """
    )


# =========================================================
# TAB 2. COIL DETAIL
# =========================================================
with tab2:
    st.subheader("Coil-Level Analysis")

    detail_cols = [
        "鍍製別",
        "上鍍層",
        "訂單號碼",
        "產出鋼捲號碼",
        "鍍層目標值",
        "鍍層下限值",
        "北側鍍層值",
        "中央鍍層值",
        "南側鍍層值",
        "鋼捲平均鍍層值",
        "目標差異",
        "目標差異率(%)",
        "下限餘裕",
        "下限餘裕率(%)",
        "最低位置",
        "最低位置鍍層值",
        "橫向最大差異",
        "橫向差異率(%)",
        "鋼捲判定",
    ]

    st.dataframe(
        format_table(filtered_df[detail_cols], 2),
        use_container_width=True,
        height=600,
    )

    abnormal_df = filtered_df[
        filtered_df["鋼捲判定"] != "符合要求"
    ].copy()

    st.subheader("Abnormal Coil List")

    if abnormal_df.empty:
        st.success("目前篩選範圍內沒有異常鋼捲。")
    else:
        st.dataframe(
            format_table(abnormal_df[detail_cols], 2),
            use_container_width=True,
            height=380,
        )

    export_bytes = dataframe_to_excel_bytes(
        detail_df=filtered_df[detail_cols],
        summary_df=summary_df[summary_display_cols],
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
# TAB 3. CHARTS
# =========================================================
with tab3:
    st.subheader("Visual Analysis")

    max_chart_rows = st.slider(
        "Maximum coils shown in charts",
        min_value=10,
        max_value=max(10, min(300, len(filtered_df))),
        value=min(50, len(filtered_df)),
        step=10 if len(filtered_df) >= 10 else 1,
    )

    chart_df = filtered_df.head(max_chart_rows).copy()

    st.pyplot(
        plot_coil_actual_vs_limits(chart_df),
        use_container_width=True,
    )

    st.pyplot(
        plot_target_deviation(chart_df),
        use_container_width=True,
    )

    st.pyplot(
        plot_position_profile(chart_df),
        use_container_width=True,
    )


# =========================================================
# TAB 4. DATA QUALITY
# =========================================================
with tab4:
    st.subheader("Data Quality")

    q1, q2, q3 = st.columns(3)
    q1.metric("原始筆數", f"{len(raw_df):,}")
    q2.metric("有效筆數", f"{len(valid_df):,}")
    q3.metric("排除筆數", f"{len(rejected_df):,}")

    if rejected_df.empty:
        st.success("沒有被排除的資料。")
    else:
        rejected_cols = [
            column
            for column in [
                "鍍製別",
                "上鍍層",
                "訂單號碼",
                "產出鋼捲號碼",
                "鍍層目標值",
                "鍍層下限值",
                "Reject_Reason",
            ]
            if column in rejected_df.columns
        ]

        st.dataframe(
            rejected_df[rejected_cols],
            use_container_width=True,
            height=500,
        )


# =========================================================
# 9. FOOTNOTE
# =========================================================
st.caption(
    "計算方式：北側=(XRAY_A_T_N+XRAY_A_B_N)/2；"
    "中央=(XRAY_A_T_C+XRAY_A_B_C)/2；"
    "南側=(XRAY_A_T_S+XRAY_A_B_S)/2；"
    "鋼捲平均=(北側+中央+南側)/3。"
)
