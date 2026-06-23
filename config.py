"""
全局配置文件 — 果园病虫害风险预警看板
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data_process", "data.csv")

# ==================== 看板基础配置 ====================
APP_TITLE = "果园病虫害风险预警与防控可视化看板"
APP_SUBTITLE = "基于 LightGBM + SHAP 的智能预警系统"
APP_LAYOUT = "wide"

# ==================== 实时刷新配置 ====================
AUTO_REFRESH_INTERVAL = 8
ENABLE_AUTO_REFRESH = False

# ==================== 风险等级配置 ====================
RISK_LEVELS = {
    0: {"label": "低风险", "color": "#2ecc71", "bg": "#eafaf1"},
    1: {"label": "中风险", "color": "#f39c12", "bg": "#fef9e7"},
    2: {"label": "高风险", "color": "#e74c3c", "bg": "#fdedec"},
}

RISK_LABEL_MAP = {0: "低风险", 1: "中风险", 2: "高风险"}
RISK_LABEL_REVERSE = {"低风险": 0, "中风险": 1, "高风险": 2, "低": 0, "中": 1, "高": 2}

# ==================== 防控方案配置 ====================
BASE_CONTROL_PLANS = {
    0: "【低风险】地块健康，执行常规巡检，无需施药。推荐巡检时段：每日上午9:00，频率：每7天1次。",
    1: "【中风险】局部发病，窗口期内点状施药。傍晚17:00喷施预防性药剂（减量），每3天1次监测，农药2种轮换(B→C)。",
    2: "【高风险】紧急！立即隔离地块，上午8:00完成全覆盖应急施药（全量），每天1次监测，农药3种轮换(A→B→C)。"
}

# ==================== 图表颜色方案 ====================
CHART_COLORS = {
    "low": "#2ecc71",
    "medium": "#f39c12",
    "high": "#e74c3c",
    "primary": "#3498db",
    "secondary": "#9b59b6",
    "accent": "#1abc9c",
    "bg_card": "#ffffff",
    "bg_page": "#f5f7fa",
    "text_primary": "#2c3e50",
    "text_secondary": "#7f8c8d",
    "border": "#e8ecf1",
}

# ==================== 图表通用配置 ====================
CHART_FONT_FAMILY = "Microsoft YaHei, SimHei, -apple-system, sans-serif"
CHART_TEMPLATE = "plotly_white"
CHART_HEIGHT_DEFAULT = 400

# ==================== 上传文件配置 ====================
ALLOWED_EXTENSIONS = ["csv"]
MAX_FILE_SIZE_MB = 50
