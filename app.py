# -*- coding: utf-8 -*-
"""
果园病虫害风险预警与防控可视化看板
=====================================
基于 LightGBM + SHAP 的智能预警系统
企业级 Streamlit 可视化大屏
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from config import (
    APP_TITLE, APP_SUBTITLE, APP_LAYOUT, AUTO_REFRESH_INTERVAL,
    ENABLE_AUTO_REFRESH, RISK_LEVELS, BASE_CONTROL_PLANS,
    OUTPUT_DIR, DEFAULT_DATA_PATH, CHART_COLORS, RISK_LABEL_MAP, RISK_LABEL_REVERSE
)
from utils.data_loader import (
    load_main_dataset, load_prevention_plan, load_response_zone_summary,
    load_three_tier_summary, load_strategy_system, load_risk_window_cross,
    load_feature_importance, load_shap_contributions, load_kpi_metrics,
    load_confusion_matrix, load_roc_auc, load_category_distribution,
    load_posi_weights, get_risk_color_map, load_rri_jenks
)
from utils.charts import (
    create_risk_pie_chart, create_risk_bar_chart, create_spatial_risk_map,
    create_time_trend_chart, create_feature_importance_chart,
    create_shap_chart, create_kpi_gauge, create_response_zone_chart,
    create_posi_weight_chart, create_risk_heatmap
)


# ==================== 页面基础配置 ====================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="",
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded"
)

# ==================== 专业 CSS 样式 ====================
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Microsoft YaHei', 'Segoe UI', -apple-system, sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f36 0%, #1e2640 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e4ec !important; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important; font-weight: 600; letter-spacing: 0.5px;
    }
    [data-testid="stSidebar"] button {
        background: #2e3a5c !important; color: #fff !important;
        border: 1px solid #3d4f7c !important; border-radius: 6px !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: #3d4f7c !important; border-color: #5a6fa8 !important;
    }
    [data-testid="stSidebar"] hr { border-color: #2e3a5c !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #8e9bb4 !important; font-size: 0.85rem;
    }

    .app-header {
        padding: 6px 0 10px 0;
        border-bottom: 1px solid #e8ecf1; margin-bottom: 18px;
    }
    .app-title {
        font-size: 1.6rem; font-weight: 700; color: #1a1f36; letter-spacing: 1px;
    }
    .app-subtitle {
        font-size: 0.85rem; color: #7f8c8d; font-weight: 400; margin-top: 2px;
    }

    .status-bar {
        display: flex; align-items: center; gap: 18px;
        padding: 8px 16px; background: #f5f7fa; border-radius: 8px;
        margin-bottom: 16px; font-size: 0.82rem; color: #546e7a;
    }
    .status-indicator {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #2ecc71; margin-right: 6px;
        box-shadow: 0 0 6px rgba(46,204,113,0.4);
    }
    .status-sep { color: #cfd8dc; }

    .kpi-grid {
        display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px;
        margin: 10px 0 20px 0;
    }
    .kpi-card {
        background: #ffffff; border-radius: 10px; padding: 18px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06); border: 1px solid #eef0f4;
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .kpi-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); transform: translateY(-2px); }
    .kpi-value { font-size: 2rem; font-weight: 700; line-height: 1.2; }
    .kpi-label { font-size: 0.8rem; color: #7f8c8d; margin-top: 4px; font-weight: 500; }
    .kpi-sub { font-size: 0.72rem; color: #b0bec5; margin-top: 2px; }
    .kpi-accent-low { color: #2ecc71; }
    .kpi-accent-mid { color: #f39c12; }
    .kpi-accent-high { color: #e74c3c; }
    .kpi-accent-primary { color: #3498db; }
    .kpi-accent-secondary { color: #9b59b6; }

    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #1a1f36;
        margin: 28px 0 12px 0; padding-bottom: 8px;
        border-bottom: 2px solid #3498db; display: inline-block; letter-spacing: 0.5px;
    }

    .advice-card {
        padding: 14px 18px; border-radius: 8px; border-left: 4px solid;
        margin: 8px 0; font-size: 0.88rem; line-height: 1.6;
    }
    .advice-low { background: #eafaf1; border-color: #2ecc71; color: #1a5c2e; }
    .advice-mid { background: #fef9e7; border-color: #f39c12; color: #7d5e0a; }
    .advice-high { background: #fdedec; border-color: #e74c3c; color: #7b241c; }

    .legend-box {
        background: #fff; border: 1px solid #eef0f4; border-radius: 8px;
        padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .legend-item {
        display: flex; align-items: center; margin: 6px 0;
        font-size: 0.85rem; color: #546e7a;
    }
    .legend-dot {
        width: 12px; height: 12px; border-radius: 3px;
        margin-right: 10px; flex-shrink: 0;
    }

    .app-footer {
        text-align: center; padding: 20px; color: #bdc3c7;
        font-size: 0.75rem; border-top: 1px solid #ecf0f1; margin-top: 40px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==================== 会话状态 ====================
if 'pipeline_ran' not in st.session_state:
    st.session_state.pipeline_ran = False
if 'pipeline_stats' not in st.session_state:
    st.session_state.pipeline_stats = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'current_data_source' not in st.session_state:
    st.session_state.current_data_source = "default"


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## Dashboard Control")
    st.markdown("---")

    # 数据源
    st.markdown("### Data Source")
    data_source = st.radio(
        "Select data source",
        ["Default Sample Data", "Upload New Dataset"],
        label_visibility="collapsed"
    )

    if data_source == "Upload New Dataset":
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Required columns: 地块ID, 果树品种, 病虫害类型, 风险等级, 平均气温, 相对湿度, 降水量, 日照时数, 土壤湿度, 风速, 近7天病株数, 近7天虫口密度, 近30天用药次数",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            uploads_dir = os.path.join(BASE_DIR, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            tmp_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_file_path = tmp_path
            st.session_state.current_data_source = "uploaded"
            st.success(f"Uploaded: {uploaded_file.name}")
    else:
        st.session_state.current_data_source = "default"

    st.markdown("---")

    # 管线
    st.markdown("### Pipeline")

    if st.session_state.current_data_source == "uploaded" and st.session_state.uploaded_file_path:
        pipeline_input = st.session_state.uploaded_file_path
        btn_label = "Run Pipeline (Uploaded Data)"
        btn_help = f"Process {os.path.basename(pipeline_input)}"
    else:
        pipeline_input = DEFAULT_DATA_PATH
        btn_label = "Run Pipeline (Sample Data)"
        btn_help = "Process built-in sample data"

    if st.button(btn_label, type="primary", width="stretch", help=btn_help):
        with st.spinner("Running data pipeline..."):
            from pipeline import run_pipeline
            result = run_pipeline(pipeline_input)
            st.session_state.pipeline_ran = True
            st.session_state.pipeline_stats = result
            if result['success']:
                st.success(f"Pipeline completed ({result['stats'].get('elapsed', 'N/A')})")
                st.cache_data.clear()
            else:
                st.error(f"Pipeline failed: {result['message']}")

    if st.session_state.pipeline_ran and st.session_state.pipeline_stats:
        s = st.session_state.pipeline_stats.get('stats', {})
        if st.session_state.pipeline_stats['success']:
            st.caption(f"Samples: {s.get('n_samples','?')} | CV-F1: {s.get('cv_macro_f1','?')} | AUC: {s.get('macro_auc','?')}")

    st.markdown("---")

    # 筛选
    st.markdown("### Filters")

    @st.cache_data(ttl=30)
    def _get_filter_options():
        df = load_main_dataset()
        v = ["全部"] + sorted(df["果树品种"].unique().tolist()) if "果树品种" in df.columns else ["全部"]
        p = ["全部"] + sorted(df["病虫害类型"].unique().astype(str).tolist()) if "病虫害类型" in df.columns else ["全部"]
        return v, p

    varieties, pests = _get_filter_options()
    selected_variety = st.selectbox("Variety", varieties, label_visibility="collapsed")
    selected_pest = st.selectbox("Pest Type", pests, label_visibility="collapsed")
    risk_filter = st.multiselect(
        "Risk Level",
        ["低风险", "中风险", "高风险"],
        default=["低风险", "中风险", "高风险"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### System Info")
    df_chk = load_main_dataset()
    st.caption(f"Total plots: {len(df_chk)}")
    if "预测风险标签" in df_chk.columns:
        rc = df_chk["预测风险标签"].value_counts()
        for lvl in ["低", "中", "高"]:
            st.caption(f"Risk {lvl}: {rc.get(lvl, 0)}")
    st.caption(f"Updated: {datetime.datetime.now().strftime('%H:%M:%S')}")


# ==================== 主内容区 ====================

# 头部
st.markdown(f"""
<div class="app-header">
    <div class="app-title">{APP_TITLE}</div>
    <div class="app-subtitle">{APP_SUBTITLE}</div>
</div>
""", unsafe_allow_html=True)

# 状态栏
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
data_label = "Uploaded" if st.session_state.current_data_source == "uploaded" else "Default"

df_main = load_main_dataset()
if df_main.empty:
    st.warning("No data found. Please click **Run Pipeline** in the sidebar to generate data, or upload a new dataset.")
    st.markdown("""
    ### Quick Start
    1. Choose a data source from the sidebar
    2. Click **Run Pipeline** to process
    3. Explore interactive visualizations
    """)
    st.stop()

st.markdown(f"""
<div class="status-bar">
    <span><span class="status-indicator"></span> System Active</span>
    <span class="status-sep">|</span>
    <span>Time: {current_time}</span>
    <span class="status-sep">|</span>
    <span>Source: {data_label}</span>
    <span class="status-sep">|</span>
    <span>Plots: {len(df_main)}</span>
</div>
""", unsafe_allow_html=True)


# ==================== 数据筛选 ====================
df_filtered = df_main.copy()

risk_code_map = {"低风险": 0, "中风险": 1, "高风险": 2, "低": 0, "中": 1, "高": 2}
if selected_variety != "全部" and "果树品种" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["果树品种"] == selected_variety]
if selected_pest != "全部" and "病虫害类型" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["病虫害类型"].astype(str) == selected_pest]

if "风险等级编码" in df_filtered.columns:
    allowed = [risk_code_map.get(r, r) for r in risk_filter]
    df_filtered = df_filtered[df_filtered["风险等级编码"].isin(allowed)]
elif "预测风险标签" in df_filtered.columns:
    allowed_labels = [r.replace("风险", "") for r in risk_filter]
    df_filtered = df_filtered[df_filtered["预测风险标签"].isin(allowed_labels)]

# 风险统计
if "风险等级编码" in df_filtered.columns:
    risk_col = "风险等级编码"
    risk_label_col = "预测风险标签" if "预测风险标签" in df_filtered.columns else None
elif "预测风险标签" in df_filtered.columns:
    df_filtered["_rc"] = df_filtered["预测风险标签"].map(lambda x: {"低": 0, "中": 1, "高": 2}.get(str(x), -1))
    risk_col = "_rc"
    risk_label_col = "预测风险标签"
else:
    risk_col = None
    risk_label_col = None

if risk_col:
    rc = df_filtered[risk_col].value_counts().to_dict()
    low_count = rc.get(0, 0)
    mid_count = rc.get(1, 0)
    high_count = rc.get(2, 0)
else:
    low_count = mid_count = high_count = 0

total = len(df_filtered)

if "在防治窗口内" in df_filtered.columns:
    wc = df_filtered["在防治窗口内"]
    window_count = int((wc == True).sum() + (wc == "True").sum() + (wc == 1).sum())
else:
    window_count = high_count


# ==================== KPI 指标卡 ====================
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-value kpi-accent-low">{low_count}</div>
        <div class="kpi-label">Low Risk</div>
        <div class="kpi-sub">{low_count/max(total,1)*100:.1f}% of total</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value kpi-accent-mid">{mid_count}</div>
        <div class="kpi-label">Medium Risk</div>
        <div class="kpi-sub">{mid_count/max(total,1)*100:.1f}% of total</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value kpi-accent-high">{high_count}</div>
        <div class="kpi-label">High Risk</div>
        <div class="kpi-sub">{high_count/max(total,1)*100:.1f}% of total</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value kpi-accent-primary">{total}</div>
        <div class="kpi-label">Total Plots</div>
        <div class="kpi-sub">After filtering</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value kpi-accent-secondary">{window_count}</div>
        <div class="kpi-label">Prevention Window</div>
        <div class="kpi-sub">Requires action</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==================== 模块1: 风险态势总览 ====================
st.markdown('<div class="section-title">Risk Overview</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    pie_fig = create_risk_pie_chart(df_filtered)
    st.plotly_chart(pie_fig, width="stretch")
with c2:
    bar_fig = create_risk_bar_chart(df_filtered)
    st.plotly_chart(bar_fig, width="stretch")


# ==================== 模块2: 空间风险分布 ====================
st.markdown('<div class="section-title">Spatial Risk Distribution</div>', unsafe_allow_html=True)

m1, m2 = st.columns([2, 1])
with m1:
    map_fig = create_spatial_risk_map(df_filtered)
    st.plotly_chart(map_fig, width="stretch")
with m2:
    st.markdown("""
    <div class="legend-box">
        <div style="font-weight:700;color:#1a1f36;margin-bottom:10px;">Risk Legend</div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#2ecc71;"></div>
            <span>Low Risk &mdash; Routine monitoring</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#f39c12;"></div>
            <span>Medium Risk &mdash; Preventive spray</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#e74c3c;"></div>
            <span>High Risk &mdash; Emergency response</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "果树品种" in df_filtered.columns and risk_col:
        st.markdown("#### Variety Breakdown")
        try:
            vb = df_filtered.groupby("果树品种")[risk_col].value_counts().unstack(fill_value=0)
            vb.columns = [RISK_LABEL_MAP.get(c, f"Lv.{c}") for c in vb.columns]
            st.dataframe(vb, width="stretch")
        except Exception:
            st.caption("Insufficient data")


# ==================== 模块3: 趋势与交叉分析 ====================
st.markdown('<div class="section-title">Trend &amp; Cross Analysis</div>', unsafe_allow_html=True)

t1, t2 = st.columns(2)
with t1:
    trend_fig = create_time_trend_chart(df_filtered)
    st.plotly_chart(trend_fig, width="stretch")
with t2:
    heatmap_fig = create_risk_heatmap(df_filtered)
    st.plotly_chart(heatmap_fig, width="stretch")

st.markdown("#### Risk Level x Prevention Window")
if "在防治窗口内" in df_filtered.columns:
    df_filtered["_win"] = df_filtered["在防治窗口内"].apply(
        lambda x: "In Window" if str(x).lower() in ["true", "1", "是"] else "Outside"
    )
    rl = risk_label_col if risk_label_col else risk_col
    cross = pd.crosstab(df_filtered[rl], df_filtered["_win"], margins=True, margins_name="Total")
    st.dataframe(cross, width="stretch")
else:
    st.caption("Prevention window data not available.")


# ==================== 模块4: 模型性能 ====================
st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

try:
    kpi_df = load_kpi_metrics()
    roc_df = load_roc_auc()
    cm_df = load_confusion_matrix()
    feat_df = load_feature_importance()

    p1, p2, p3 = st.columns(3)
    with p1:
        if not kpi_df.empty:
            st.dataframe(kpi_df, width="stretch", hide_index=True)
    with p2:
        if not roc_df.empty:
            st.dataframe(roc_df, width="stretch", hide_index=True)
    with p3:
        if not cm_df.empty:
            st.dataframe(cm_df, width="stretch")

    st.markdown("#### Feature Importance (LightGBM Gain)")
    if not feat_df.empty:
        fi_fig = create_feature_importance_chart(feat_df)
        st.plotly_chart(fi_fig, width="stretch")

    shap_df = load_shap_contributions()
    if not shap_df.empty:
        st.markdown("#### SHAP Contribution")
        shap_fig = create_shap_chart(shap_df)
        st.plotly_chart(shap_fig, width="stretch")

except Exception as e:
    st.info(f"Model performance data unavailable: {e}")


# ==================== 模块5: 防控策略 ====================
st.markdown('<div class="section-title">Prevention Strategy</div>', unsafe_allow_html=True)

try:
    rri_df = load_rri_jenks()
    posi_df = load_posi_weights()
    prev_df = load_prevention_plan()

    s1, s2 = st.columns(2)
    with s1:
        if not rri_df.empty:
            zone_fig = create_response_zone_chart(rri_df)
            st.plotly_chart(zone_fig, width="stretch")
    with s2:
        if not posi_df.empty:
            posi_fig = create_posi_weight_chart(posi_df)
            st.plotly_chart(posi_fig, width="stretch")

    st.markdown("#### Prevention Plans")
    if not prev_df.empty:
        st.dataframe(prev_df.head(20), width="stretch")
except Exception as e:
    st.info(f"Strategy data unavailable: {e}")


# ==================== 模块6: 地块详情 ====================
st.markdown('<div class="section-title">Plot Detail</div>', unsafe_allow_html=True)

if "地块ID" not in df_filtered.columns:
    df_filtered["地块ID"] = [f"P{i+1}" for i in range(len(df_filtered))]

plot_ids = df_filtered["地块ID"].tolist()

f1, f2 = st.columns([1, 3])
with f1:
    quick = st.radio("Quick Filter", ["All", "High Risk", "Medium Risk", "Low Risk"],
                     label_visibility="collapsed")
with f2:
    if quick == "High Risk" and risk_col:
        candidate = df_filtered[df_filtered[risk_col] == 2]["地块ID"].tolist()
    elif quick == "Medium Risk" and risk_col:
        candidate = df_filtered[df_filtered[risk_col] == 1]["地块ID"].tolist()
    elif quick == "Low Risk" and risk_col:
        candidate = df_filtered[df_filtered[risk_col] == 0]["地块ID"].tolist()
    else:
        candidate = plot_ids

    if candidate:
        selected_plot = st.selectbox("Select a plot", candidate, label_visibility="collapsed")
        plot_data = df_filtered[df_filtered["地块ID"] == selected_plot]
        if not plot_data.empty:
            row = plot_data.iloc[0]
            risk_val = row.get("预测风险标签", row.get(risk_col, "N/A"))
            risk_code_val = {"低": 0, "中": 1, "高": 2}.get(str(risk_val), 0)
            advice = BASE_CONTROL_PLANS.get(risk_code_val, "No specific plan.")
            css_map = {0: "advice-low", 1: "advice-mid", 2: "advice-high"}
            css_class = css_map.get(risk_code_val, "advice-low")

            st.markdown(f"""
            <div class="advice-card {css_class}">
                <strong>Plot: {selected_plot}</strong> &nbsp;|&nbsp; Risk: {risk_val}<br>
                {advice}
            </div>
            """, unsafe_allow_html=True)

            display_cols = [c for c in df_filtered.columns if c not in ['_rc', '_win']
                            and not c.startswith('_')][:15]
            st.dataframe(plot_data[display_cols], width="stretch")
    else:
        st.info("No plots match current filters.")


# ==================== 页脚 ====================
st.markdown(f"""
<div class="app-footer">
    Orchard Pest Risk Warning Dashboard &middot; Powered by LightGBM &middot; {datetime.datetime.now().year}
</div>
""", unsafe_allow_html=True)
