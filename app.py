# -*- coding: utf-8 -*-
"""
果园病虫害风险预警与防控可视化看板
=====================================
基于 LightGBM + SHAP 的智能预警系统
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from config import (
    APP_TITLE, APP_SUBTITLE, APP_LAYOUT,
    RISK_LEVELS, BASE_CONTROL_PLANS,
    OUTPUT_DIR, DEFAULT_DATA_PATH, RISK_LABEL_MAP, RISK_LABEL_REVERSE
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
    create_shap_chart, create_response_zone_chart,
    create_posi_weight_chart, create_risk_heatmap
)


# ==================== 页面配置 ====================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="",
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded"
)

# ==================== CSS 样式 ====================
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
        color: #ffffff !important; font-weight: 600;
    }
    [data-testid="stSidebar"] button {
        background: #2e3a5c !important; color: #fff !important;
        border: 1px solid #3d4f7c !important; border-radius: 6px !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: #3d4f7c !important; border-color: #5a6fa8 !important;
    }
    [data-testid="stSidebar"] hr { border-color: #2e3a5c !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #e0e4ec !important;
    }
    [data-testid="stSidebar"] .st-emotion-cache-1qg05tj {
        color: #8e9bb4 !important; font-size: 0.82rem;
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
        display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
        padding: 8px 16px; background: #f5f7fa; border-radius: 8px;
        margin-bottom: 16px; font-size: 0.82rem; color: #546e7a;
    }
    .status-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #2ecc71; margin-right: 6px;
        box-shadow: 0 0 6px rgba(46,204,113,0.4);
    }
    .status-dot-warn {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #f39c12; margin-right: 6px;
        box-shadow: 0 0 6px rgba(243,156,18,0.4);
    }

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

    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #1a1f36;
        margin: 28px 0 12px 0; padding-bottom: 8px;
        border-bottom: 2px solid #3498db; display: inline-block;
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
</style>
""", unsafe_allow_html=True)


# ==================== 会话状态 ====================
for key, default in [
    ('pipeline_done', False),
    ('pipeline_stats', None),
    ('uploaded_path', None),
    ('using_uploaded', False),
    ('pipeline_running', False),
    ('upload_valid', False),
    ('upload_errors', []),
    ('upload_preview', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# 数据格式要求
REQUIRED_COLUMNS = [
    '地块ID', '果树品种', '病虫害类型', '风险等级',
    '平均气温', '相对湿度', '降水量', '日照时数', '土壤湿度', '风速',
    '近7天病株数', '近7天虫口密度', '近30天用药次数'
]


def validate_uploaded_csv(file_path: str) -> tuple:
    """
    校验上传的 CSV 是否符合管线所需格式
    返回: (是否有效, 错误列表, 预览DataFrame)
    """
    errors = []
    df = None

    # 尝试读取
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030']:
        try:
            df = pd.read_csv(file_path, encoding=enc, nrows=5)
            break
        except Exception:
            continue

    if df is None:
        return False, ["无法读取文件，请确认文件为有效的 CSV 格式（编码：UTF-8 或 GBK）"], None

    # 检查必填列
    existing = set(df.columns)
    required = set(REQUIRED_COLUMNS)
    missing = required - existing

    if missing:
        errors.append(f"缺少 {len(missing)} 个必填字段：{', '.join(sorted(missing))}")

    # 检查风险等级值域
    if '风险等级' in existing:
        valid_levels = {'低', '中', '高'}
        actual = set(df['风险等级'].dropna().unique())
        invalid = actual - valid_levels
        if invalid:
            errors.append(f"「风险等级」列包含非法值 {invalid}，仅接受：低、中、高")

    # 检查数值列
    numeric_check_cols = ['平均气温', '相对湿度', '降水量', '日照时数', '土壤湿度', '风速',
                          '近7天病株数', '近7天虫口密度', '近30天用药次数']
    for col in numeric_check_cols:
        if col in existing:
            try:
                pd.to_numeric(df[col])
            except Exception:
                errors.append(f"「{col}」列包含非数值数据，请检查")

    valid = len(errors) == 0
    return valid, errors, df.head(3) if df is not None else None


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## 控制面板")
    st.markdown("---")

    # ---------- 第一步：上传数据 ----------
    st.markdown("### 第一步：选择数据源")

    data_mode = st.radio(
        "数据来源",
        ["使用示例数据", "上传新数据集"],
        label_visibility="collapsed"
    )

    if data_mode == "上传新数据集":
        uploaded_file = st.file_uploader(
            "选择 CSV 文件上传",
            type=["csv"],
            help="需含以下必填列：地块ID, 果树品种, 病虫害类型, 风险等级, 平均气温, 相对湿度, 降水量, 日照时数, 土壤湿度, 风速, 近7天病株数, 近7天虫口密度, 近30天用药次数",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            # 存到项目 uploads/ 目录
            uploads_dir = os.path.join(BASE_DIR, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            save_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 校验格式
            valid, errors, preview = validate_uploaded_csv(save_path)
            st.session_state.uploaded_path = save_path
            st.session_state.using_uploaded = True
            st.session_state.upload_valid = valid
            st.session_state.upload_errors = errors
            st.session_state.upload_preview = preview
            st.session_state.pipeline_done = False

            # 显示存储路径
            st.caption(f"存储位置：`uploads/{uploaded_file.name}`")

            if valid:
                st.success(f"格式校验通过 — {uploaded_file.name}")
                st.info("请点击下方「开始分析」")
            else:
                st.error("格式校验未通过，无法执行分析：")
                for err in errors:
                    st.warning(f"- {err}")

            # 预览数据
            if preview is not None:
                with st.expander("数据预览（前3行）"):
                    st.dataframe(preview, width="stretch")
    else:
        st.session_state.using_uploaded = False
        st.session_state.upload_valid = False
        st.caption("当前数据：内置示例 (300条)")

    st.markdown("---")

    # ---------- 第二步：运行分析 ----------
    st.markdown("### 第二步：运行分析")

    if st.session_state.using_uploaded and st.session_state.uploaded_path:
        pipe_input = st.session_state.uploaded_path
        btn_text = "开始分析（上传数据）"
        can_run = st.session_state.upload_valid
    else:
        pipe_input = DEFAULT_DATA_PATH
        btn_text = "开始分析（示例数据）"
        can_run = True

    if not can_run:
        st.warning("上传数据格式不符，请修正后重新上传")
        btn_disabled = True
    else:
        btn_disabled = False

    if st.button(btn_text, type="primary", width="stretch", disabled=btn_disabled):
        st.session_state.pipeline_running = True
        st.session_state.pipeline_done = False

    if st.session_state.pipeline_running:
        status_placeholder = st.empty()
        try:
            status_placeholder.info("正在执行数据管线，请稍候...")
            from pipeline import run_pipeline
            result = run_pipeline(pipe_input)
            st.session_state.pipeline_stats = result
            st.session_state.pipeline_running = False

            if result['success']:
                st.session_state.pipeline_done = True
                st.cache_data.clear()
                status_placeholder.success(
                    f"分析完成！耗时 {result['stats'].get('elapsed','?')}"
                )
                st.rerun()
            else:
                status_placeholder.error(f"分析失败：{result['message']}")
        except Exception as e:
            st.session_state.pipeline_running = False
            status_placeholder.error(f"管线异常：{str(e)}")

    if st.session_state.pipeline_done and st.session_state.pipeline_stats:
        s = st.session_state.pipeline_stats.get('stats', {})
        if st.session_state.pipeline_stats.get('success'):
            st.caption(
                f"样本数：{s.get('n_samples','?')} | "
                f"F1分数：{s.get('cv_macro_f1','?')} | "
                f"AUC：{s.get('macro_auc','?')}"
            )

    st.markdown("---")

    # ---------- 第三步：筛选查看 ----------
    st.markdown("### 第三步：筛选查看")

    @st.cache_data(ttl=30)
    def _get_filter_options():
        df = load_main_dataset()
        v = ["全部"] + sorted(df["果树品种"].unique().tolist()) if "果树品种" in df.columns else ["全部"]
        p = ["全部"] + sorted(df["病虫害类型"].unique().astype(str).tolist()) if "病虫害类型" in df.columns else ["全部"]
        return v, p

    varieties, pests = _get_filter_options()
    selected_variety = st.selectbox("果树品种", varieties, label_visibility="collapsed")
    selected_pest = st.selectbox("病虫害类型", pests, label_visibility="collapsed")
    risk_filter = st.multiselect(
        "风险等级",
        ["低风险", "中风险", "高风险"],
        default=["低风险", "中风险", "高风险"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # ---------- 系统信息 ----------
    st.markdown("### 系统信息")
    df_chk = load_main_dataset()
    st.caption(f"地块总数：{len(df_chk)}")
    if "预测风险标签" in df_chk.columns:
        rc = df_chk["预测风险标签"].value_counts()
        for lvl in ["低", "中", "高"]:
            st.caption(f"{lvl}风险：{rc.get(lvl, 0)} 块")
    st.caption(f"更新时间：{datetime.datetime.now().strftime('%H:%M:%S')}")


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
source_label = "上传数据" if st.session_state.using_uploaded else "示例数据"

df_main = load_main_dataset()
if df_main.empty:
    st.warning("暂无数据，请先在左侧面板上传数据并点击「开始分析」按钮。")
    st.markdown("""
    ### 操作指引
    1. **选择数据源**：可以使用内置示例，或上传您自己的 CSV 数据集
    2. **点击「开始分析」**：系统将自动运行数据处理管线（预处理 → 建模 → 防控方案）
    3. **查看可视化结果**：分析完成后即可查看完整的风险预警看板
    """)
    st.stop()

# 如果刚完成管线，清缓存后需要 rerun 已处理；如果没有则正常展示
pipeline_ok = (st.session_state.pipeline_done and
               st.session_state.pipeline_stats and
               st.session_state.pipeline_stats.get('success'))

st.markdown(f"""
<div class="status-bar">
    <span><span class="status-dot"></span> 系统运行中</span>
    <span>时间：{current_time}</span>
    <span>数据源：{source_label}</span>
    <span>地块数：{len(df_main)}</span>
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
    df_filtered["_rc"] = df_filtered["预测风险标签"].map(
        lambda x: {"低": 0, "中": 1, "高": 2}.get(str(x), -1))
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
        <div class="kpi-value" style="color:#2ecc71;">{low_count}</div>
        <div class="kpi-label">低风险地块</div>
        <div class="kpi-sub">占比 {low_count/max(total,1)*100:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="color:#f39c12;">{mid_count}</div>
        <div class="kpi-label">中风险地块</div>
        <div class="kpi-sub">占比 {mid_count/max(total,1)*100:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="color:#e74c3c;">{high_count}</div>
        <div class="kpi-label">高风险地块</div>
        <div class="kpi-sub">占比 {high_count/max(total,1)*100:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="color:#3498db;">{total}</div>
        <div class="kpi-label">筛选后地块总数</div>
        <div class="kpi-sub">当前筛选条件</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="color:#9b59b6;">{window_count}</div>
        <div class="kpi-label">防治窗口内地块</div>
        <div class="kpi-sub">需要立即处置</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==================== 模块1：风险态势总览 ====================
st.markdown('<div class="section-title">一、全园风险态势总览</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    pie_fig = create_risk_pie_chart(df_filtered)
    st.plotly_chart(pie_fig, width="stretch")
with c2:
    bar_fig = create_risk_bar_chart(df_filtered)
    st.plotly_chart(bar_fig, width="stretch")


# ==================== 模块2：空间风险分布 ====================
st.markdown('<div class="section-title">二、果园分区域风险空间分布</div>', unsafe_allow_html=True)

m1, m2 = st.columns([2, 1])
with m1:
    map_fig = create_spatial_risk_map(df_filtered)
    st.plotly_chart(map_fig, width="stretch")
with m2:
    st.markdown("""
    <div class="legend-box">
        <div style="font-weight:700;color:#1a1f36;margin-bottom:10px;">风险等级图例</div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#2ecc71;"></div>
            <span>低风险 — 常规监测，无需施药</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#f39c12;"></div>
            <span>中风险 — 预防施药，加强巡检</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:#e74c3c;"></div>
            <span>高风险 — 应急响应，立即处置</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "果树品种" in df_filtered.columns and risk_col:
        st.markdown("#### 品种风险分布")
        try:
            vb = df_filtered.groupby("果树品种")[risk_col].value_counts().unstack(fill_value=0)
            vb.columns = [RISK_LABEL_MAP.get(c, f"等级{c}") for c in vb.columns]
            st.dataframe(vb, width="stretch")
        except Exception:
            st.caption("数据不足")


# ==================== 模块3：趋势与交叉分析 ====================
st.markdown('<div class="section-title">三、分时段风险趋势与交叉分析</div>', unsafe_allow_html=True)

t1, t2 = st.columns(2)
with t1:
    trend_fig = create_time_trend_chart(df_filtered)
    st.plotly_chart(trend_fig, width="stretch")
with t2:
    heatmap_fig = create_risk_heatmap(df_filtered)
    st.plotly_chart(heatmap_fig, width="stretch")

st.markdown("#### 风险等级 x 防治窗口交叉统计")
if "在防治窗口内" in df_filtered.columns:
    df_filtered["_win"] = df_filtered["在防治窗口内"].apply(
        lambda x: "窗口内" if str(x).lower() in ["true", "1", "是"] else "窗口外"
    )
    rl = risk_label_col if risk_label_col else risk_col
    cross = pd.crosstab(df_filtered[rl], df_filtered["_win"], margins=True, margins_name="合计")
    st.dataframe(cross, width="stretch")
else:
    st.caption("当前数据集无防治窗口字段")


# ==================== 模块4：模型性能 ====================
st.markdown('<div class="section-title">四、模型性能评估</div>', unsafe_allow_html=True)

try:
    kpi_df = load_kpi_metrics()
    roc_df = load_roc_auc()
    cm_df = load_confusion_matrix()
    feat_df = load_feature_importance()

    p1, p2, p3 = st.columns(3)
    with p1:
        if not kpi_df.empty:
            st.markdown("**核心指标**")
            st.dataframe(kpi_df, width="stretch", hide_index=True)
    with p2:
        if not roc_df.empty:
            st.markdown("**ROC/AUC**")
            st.dataframe(roc_df, width="stretch", hide_index=True)
    with p3:
        if not cm_df.empty:
            st.markdown("**混淆矩阵**")
            st.dataframe(cm_df, width="stretch")

    st.markdown("#### 特征重要性（LightGBM Gain）")
    if not feat_df.empty:
        fi_fig = create_feature_importance_chart(feat_df)
        st.plotly_chart(fi_fig, width="stretch")

    shap_df = load_shap_contributions()
    if not shap_df.empty:
        st.markdown("#### SHAP 特征贡献分析")
        shap_fig = create_shap_chart(shap_df)
        st.plotly_chart(shap_fig, width="stretch")

except Exception as e:
    st.info(f"模型性能数据加载失败：{e}")


# ==================== 模块5：防控策略 ====================
st.markdown('<div class="section-title">五、精准防控策略体系</div>', unsafe_allow_html=True)

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

    st.markdown("#### 防控方案推荐列表")
    if not prev_df.empty:
        st.dataframe(prev_df.head(20), width="stretch")
except Exception as e:
    st.info(f"防控策略数据加载失败：{e}")


# ==================== 模块6：单地块详情 ====================
st.markdown('<div class="section-title">六、单地块详情与防控建议</div>', unsafe_allow_html=True)

if "地块ID" not in df_filtered.columns:
    df_filtered["地块ID"] = [f"P{i+1}" for i in range(len(df_filtered))]

plot_ids = df_filtered["地块ID"].tolist()

f1, f2 = st.columns([1, 3])
with f1:
    quick = st.radio(
        "快速筛选",
        ["全部地块", "仅高风险", "仅中风险", "仅低风险"],
        label_visibility="collapsed"
    )
with f2:
    if quick == "仅高风险" and risk_col:
        candidate = df_filtered[df_filtered[risk_col] == 2]["地块ID"].tolist()
    elif quick == "仅中风险" and risk_col:
        candidate = df_filtered[df_filtered[risk_col] == 1]["地块ID"].tolist()
    elif quick == "仅低风险" and risk_col:
        candidate = df_filtered[df_filtered[risk_col] == 0]["地块ID"].tolist()
    else:
        candidate = plot_ids

    if candidate:
        selected_plot = st.selectbox("选择地块", candidate, label_visibility="collapsed")
        plot_data = df_filtered[df_filtered["地块ID"] == selected_plot]
        if not plot_data.empty:
            row = plot_data.iloc[0]
            risk_val = row.get("预测风险标签", row.get(risk_col, "未知"))
            risk_code_val = {"低": 0, "中": 1, "高": 2}.get(str(risk_val), 0)
            advice = BASE_CONTROL_PLANS.get(risk_code_val, "暂无特定方案。")
            css_map = {0: "advice-low", 1: "advice-mid", 2: "advice-high"}
            css_class = css_map.get(risk_code_val, "advice-low")

            st.markdown(f"""
            <div class="advice-card {css_class}">
                <strong>地块：{selected_plot}</strong> &nbsp;|&nbsp; 风险等级：{risk_val}<br>
                {advice}
            </div>
            """, unsafe_allow_html=True)

            display_cols = [c for c in df_filtered.columns
                            if c not in ['_rc', '_win'] and not c.startswith('_')][:15]
            st.dataframe(plot_data[display_cols], width="stretch")
    else:
        st.info("当前筛选条件下没有匹配的地块。")


# ==================== 页脚 ====================
st.markdown(f"""
<div class="app-footer">
    果园病虫害风险预警可视化看板 &middot; 基于 LightGBM 智能分析引擎 &middot; {datetime.datetime.now().year}
</div>
""", unsafe_allow_html=True)
