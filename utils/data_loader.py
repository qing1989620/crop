"""
数据加载模块 - 统一管理所有CSV数据的读取与缓存
"""
import os
import pandas as pd
import streamlit as st

# ==================== 自适应路径（兼容本地开发 + 云端部署） ====================
# 当前文件：utils/data_loader.py
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))       # → utils/
_BASE_DIR = os.path.dirname(_CURRENT_FILE_DIR)                       # → 项目根（部署后）或 4.可视化前后端/（本地）

# 智能查找 output/ 目录：
#   部署后：output/ 与 app.py 同级 → 在 _BASE_DIR 下
#   本地：  output/ 在项目根目录      → 在 _BASE_DIR 的上一级
def _find_project_root(base_dir: str) -> str:
    """向上查找包含 output/ 目录的项目根"""
    search_dir = base_dir
    for _ in range(3):
        if os.path.exists(os.path.join(search_dir, "output")):
            return search_dir
        parent = os.path.dirname(search_dir)
        if parent == search_dir:  # 已到文件系统根
            break
        search_dir = parent
    return base_dir  # fallback

PROJECT_ROOT = _find_project_root(_BASE_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def _read_csv_safe(relative_path: str, **kwargs) -> pd.DataFrame:
    """安全读取CSV，自动处理编码问题"""
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    if not os.path.exists(full_path):
        return pd.DataFrame()
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030']:
        try:
            df = pd.read_csv(full_path, encoding=enc, **kwargs)
            return df
        except (UnicodeDecodeError, Exception):
            continue
    return pd.read_csv(full_path, **kwargs)


@st.cache_data(ttl=30)
def load_main_dataset() -> pd.DataFrame:
    """加载全量地块风险概率与标签数据（主数据集）"""
    df = _read_csv_safe("output/3.分区域分时段/tables/00_全量地块风险概率与标签.csv")
    return df


@st.cache_data(ttl=30)
def load_intuitive_features() -> pd.DataFrame:
    """加载直观特征集"""
    df = _read_csv_safe("output/1.预处理及特征工程结果/06_论文制表专用_直观特征集.csv")
    return df


@st.cache_data(ttl=60)
def load_prevention_plan() -> pd.DataFrame:
    """加载防控单元精准方案推荐"""
    df = _read_csv_safe("output/3.分区域分时段/tables/05_防控单元精准方案推荐.csv")
    return df


@st.cache_data(ttl=60)
def load_prevention_plan_compact() -> pd.DataFrame:
    """加载防控单元精准方案推荐（精简版）"""
    df = _read_csv_safe("output/3.分区域分时段/tables/表3-6_防控单元精准方案推荐_精简版.csv")
    return df


@st.cache_data(ttl=60)
def load_response_zone_summary() -> pd.DataFrame:
    """加载防控响应区汇总统计"""
    df = _read_csv_safe("output/3.分区域分时段/tables/06_防控响应区汇总统计.csv")
    return df


@st.cache_data(ttl=60)
def load_three_tier_summary() -> pd.DataFrame:
    """加载三级防控响应区汇总"""
    df = _read_csv_safe("output/3.分区域分时段/tables/表3-1_三级防控响应区汇总.csv")
    return df


@st.cache_data(ttl=60)
def load_strategy_system() -> pd.DataFrame:
    """加载三级差异化防控策略体系"""
    df = _read_csv_safe("output/3.分区域分时段/tables/表3-5_三级差异化防控策略体系.csv")
    return df


@st.cache_data(ttl=60)
def load_risk_window_cross() -> pd.DataFrame:
    """加载风险等级与防治窗口交叉统计"""
    df = _read_csv_safe("output/3.分区域分时段/tables/表3-4_风险等级与防治窗口交叉统计.csv")
    return df


@st.cache_data(ttl=60)
def load_posi_weights() -> pd.DataFrame:
    """加载POSI因子权重"""
    df = _read_csv_safe("output/3.分区域分时段/tables/08_POSI因子权重.csv")
    return df


@st.cache_data(ttl=60)
def load_feature_importance() -> pd.DataFrame:
    """加载特征重要性（LightGBM Gain）"""
    df = _read_csv_safe("output/2.低中高/tables/特征重要性.csv")
    return df


@st.cache_data(ttl=60)
def load_shap_contributions() -> pd.DataFrame:
    """加载SHAP特征贡献"""
    df = _read_csv_safe("output/2.低中高/tables/SHAP特征贡献.csv")
    return df


@st.cache_data(ttl=60)
def load_kpi_metrics() -> pd.DataFrame:
    """加载核心KPI指标"""
    df = _read_csv_safe("output/2.低中高/tables/核心KPI指标.csv")
    return df


@st.cache_data(ttl=60)
def load_confusion_matrix() -> pd.DataFrame:
    """加载混淆矩阵"""
    df = _read_csv_safe("output/2.低中高/tables/混淆矩阵.csv")
    return df


@st.cache_data(ttl=60)
def load_roc_auc() -> pd.DataFrame:
    """加载ROC AUC值"""
    df = _read_csv_safe("output/2.低中高/tables/ROC_AUC值.csv")
    return df


@st.cache_data(ttl=60)
def load_category_distribution() -> pd.DataFrame:
    """加载类别分布"""
    df = _read_csv_safe("output/2.低中高/tables/类别分布.csv")
    return df


@st.cache_data(ttl=60)
def load_cv_results() -> pd.DataFrame:
    """加载五折交叉验证结果"""
    df = _read_csv_safe("output/2.低中高/tables/CV_五折交叉验证.csv")
    return df


@st.cache_data(ttl=60)
def load_plot_details() -> pd.DataFrame:
    """加载地块级详情（前50）"""
    df = _read_csv_safe("output/3.分区域分时段/tables/07_地块级详情_前50.csv")
    return df


@st.cache_data(ttl=60)
def load_pareto_frontier() -> pd.DataFrame:
    """加载帕累托前沿数据"""
    df = _read_csv_safe("output/3.分区域分时段/tables/04_帕累托前沿_减药增效权衡.csv")
    return df


@st.cache_data(ttl=60)
def load_youden_curve() -> pd.DataFrame:
    """加载Youden标定曲线"""
    df = _read_csv_safe("output/3.分区域分时段/tables/03_POSI阈值Youden标定曲线.csv")
    return df


@st.cache_data(ttl=60)
def load_rri_jenks() -> pd.DataFrame:
    """加载RRI区域风险指数与Jenks分区"""
    df = _read_csv_safe("output/3.分区域分时段/tables/01_RRI区域风险指数与Jenks分区.csv")
    return df


def get_risk_color_map() -> dict:
    """风险等级颜色映射：低=绿，中=黄，高=红"""
    return {
        0: "green",
        1: "orange",   # 用橙色代替黄色更醒目
        2: "red",
        "低": "green",
        "中": "orange",
        "高": "red",
        "低风险": "green",
        "中风险": "orange",
        "高风险": "red",
    }


def get_risk_label_map() -> dict:
    """风险等级标签映射"""
    return {
        0: "低风险",
        1: "中风险",
        2: "高风险",
        "低": "低风险",
        "中": "中风险",
        "高": "高风险",
    }


# 分时段预设的防控建议映射（基于三级差异化防控策略体系）
FENSHIDUAN_PLANS = {
    0: {
        "morning": "【低风险·上午】地块状态良好。建议：常规巡检，通风光照管理，无需施药。",
        "afternoon": "【低风险·下午】维持常规监测即可，关注周边地块动态。",
        "evening": "【低风险·傍晚】当日巡检结束，记录温湿度数据，次日继续常规监测。",
        "default": "【低风险】地块健康，执行常规巡检，无需施药。推荐巡检时段：每日上午9:00。"
    },
    1: {
        "morning": "【中风险·上午】病虫害局部发生，加强巡检。建议：检查病株分布范围，标记重点区域。",
        "afternoon": "【中风险·下午】窗口期内点状施药，傍晚17:00喷施预防性药剂（减量）。",
        "evening": "【中风险·傍晚】执行预防性喷施，使用杀虫剂B→生物农药C轮换方案，每日巡检2次。",
        "default": "【中风险】局部发病，傍晚施药+加强巡检。推荐用药：减量轮换（≥2种），每3天1次监测。"
    },
    2: {
        "morning": "【高风险·上午】紧急！病害爆发风险极高。立即隔离地块，上午8:00完成应急施药（全量）。",
        "afternoon": "【高风险·下午】持续监控，检查施药效果。如有扩散迹象，扩大隔离范围。",
        "evening": "【高风险·傍晚】全天候监控，杀菌剂A→杀虫剂B→生物农药C三轮换（≥3种），每7天轮换。",
        "default": "【高风险】紧急！立即全域隔离+应急施药+24小时监控。推荐用药：全量三轮换（A→B→C），每天监测1次。"
    }
}


def get_time_based_plan(risk_level: int) -> str:
    """根据当前系统时间，返回分时段的防控建议"""
    import datetime
    hour = datetime.datetime.now().hour
    plans = FENSHIDUAN_PLANS.get(risk_level, FENSHIDUAN_PLANS[0])
    if 6 <= hour < 12:
        return plans["morning"]
    elif 12 <= hour < 17:
        return plans["afternoon"]
    elif 17 <= hour < 21:
        return plans["evening"]
    else:
        return plans["default"]
