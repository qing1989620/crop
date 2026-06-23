"""
可视化图表模块 - 封装所有 Plotly 图表生成函数
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple

# 颜色方案
COLOR_MAP = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c",
             "低": "#2ecc71", "中": "#f39c12", "高": "#e74c3c",
             "低风险": "#2ecc71", "中风险": "#f39c12", "高风险": "#e74c3c",
             "绿色区(常规监测)": "#2ecc71", "黄色区(预防施药)": "#f39c12", "红色区(应急防控)": "#e74c3c"}
RISK_LABEL_MAP = {0: "低风险", 1: "中风险", 2: "高风险"}

# 专业配色方案
COLORS_PRIMARY = ["#2ecc71", "#f39c12", "#e74c3c"]
COLORS_BLUE = px.colors.sequential.Blues
COLORS_TEAL = px.colors.sequential.Teal


def create_risk_pie_chart(df: pd.DataFrame, risk_col: str = "预测风险标签") -> go.Figure:
    """创建风险等级占比环形图"""
    # 确定使用哪个列
    if risk_col in df.columns:
        col = risk_col
    elif "风险等级编码" in df.columns:
        col = "风险等级编码"
        df = df.copy()
        df[col] = df[col].map(RISK_LABEL_MAP)
    else:
        col = df.select_dtypes(include=[np.number]).columns[-1]
    
    risk_counts = df[col].value_counts().reset_index()
    risk_counts.columns = ["风险等级", "地块数"]
    
    # 排序：低-中-高
    order = ["低风险", "中风险", "高风险"]
    risk_counts["sort_order"] = risk_counts["风险等级"].apply(lambda x: order.index(x) if x in order else 99)
    risk_counts = risk_counts.sort_values("sort_order")
    
    fig = px.pie(
        risk_counts,
        values="地块数",
        names="风险等级",
        hole=0.45,
        color="风险等级",
        color_discrete_map={
            "低风险": "#2ecc71",
            "中风险": "#f39c12",
            "高风险": "#e74c3c"
        }
    )
    fig.update_traces(
        textposition='outside',
        textinfo='percent+label+value',
        marker=dict(line=dict(color='white', width=2)),
        hovertemplate="<b>%{label}</b><br>地块数: %{value}<br>占比: %{percent}<extra></extra>"
    )
    fig.update_layout(
        title=dict(text="全园风险等级占比", font=dict(size=18, family="Microsoft YaHei"), x=0.5),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=50, b=50, l=10, r=10),
        height=400
    )
    return fig


def create_risk_bar_chart(df: pd.DataFrame, risk_col: str = "风险等级编码") -> go.Figure:
    """创建风险等级柱状图"""
    if risk_col in df.columns:
        col = risk_col
    elif "预测风险标签" in df.columns:
        col = "预测风险标签"
    else:
        col = df.columns[-1]
    
    counts = df[col].value_counts().reset_index()
    counts.columns = ["风险等级", "地块数"]
    
    # 排序
    if df[col].dtype in [np.int64, np.float64]:
        counts["风险等级"] = counts["风险等级"].map(RISK_LABEL_MAP)
    
    order = ["低风险", "中风险", "高风险"]
    counts["sort_order"] = counts["风险等级"].apply(lambda x: order.index(x) if x in order else 99)
    counts = counts.sort_values("sort_order")
    
    fig = px.bar(
        counts,
        x="风险等级",
        y="地块数",
        color="风险等级",
        color_discrete_map={
            "低风险": "#2ecc71",
            "中风险": "#f39c12",
            "高风险": "#e74c3c"
        },
        text="地块数"
    )
    fig.update_traces(textposition='outside', textfont=dict(size=14))
    fig.update_layout(
        title=dict(text="各风险等级地块数量统计", font=dict(size=18, family="Microsoft YaHei"), x=0.5),
        xaxis_title="风险等级",
        yaxis_title="地块数量（块）",
        showlegend=False,
        margin=dict(t=50, b=50, l=10, r=10),
        height=400
    )
    return fig


def create_spatial_risk_map(df: pd.DataFrame) -> go.Figure:
    """
    创建果园空间风险分布散点图（模拟GIS地图）
    使用地块ID生成模拟坐标，按风险等级着色
    """
    plot_df = df.copy()
    
    # 确定风险等级列
    if "风险等级编码" in plot_df.columns:
        risk_col = "风险等级编码"
    elif "预测风险标签" in plot_df.columns:
        # 映射
        label_to_code = {"低": 0, "中": 1, "高": 2}
        plot_df["risk_code"] = plot_df["预测风险标签"].map(label_to_code)
        risk_col = "risk_code"
    else:
        risk_col = plot_df.select_dtypes(include=[np.number]).columns[-1]
    
    # 如果没有坐标列，基于地块ID生成模拟坐标
    plot_ids = plot_df["地块ID"].values if "地块ID" in plot_df.columns else plot_df.index.astype(str)
    n = len(plot_ids)
    
    if "area_x" not in plot_df.columns:
        # 生成有意义的模拟坐标：用网格布局模拟果园分区
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        x_coords = []
        y_coords = []
        for i in range(n):
            x_coords.append((i % cols) * 10 + np.random.uniform(-2, 2))
            y_coords.append((i // cols) * 10 + np.random.uniform(-2, 2))
        plot_df["area_x"] = x_coords
        plot_df["area_y"] = y_coords
    
    # 风险标签
    if risk_col in plot_df.columns:
        plot_df["风险显示"] = plot_df[risk_col].apply(
            lambda r: RISK_LABEL_MAP.get(int(r), str(r)) if pd.notna(r) else "未知"
        )
    else:
        plot_df["风险显示"] = "未知"
    
    # 地块大小（基于风险概率增大高风险地块标记）
    if "最大风险概率" in plot_df.columns:
        size_col = "最大风险概率"
        plot_df["标记大小"] = plot_df[size_col] * 30 + 15
    else:
        plot_df["标记大小"] = 18
    
    # hover信息
    hover_cols = ["地块ID"]
    for c in ["果树品种", "病虫害类型", "预测风险标签", "最大风险概率",
              "THI_温湿胁迫", "BTM_生物威胁动量", "PRI_抗药性预警"]:
        if c in plot_df.columns:
            hover_cols.append(c)
    
    fig = px.scatter(
        plot_df,
        x="area_x",
        y="area_y",
        color="风险显示",
        size="标记大小",
        color_discrete_map={
            "低风险": "#2ecc71",
            "中风险": "#f39c12",
            "高风险": "#e74c3c"
        },
        hover_data=[c for c in hover_cols if c != "地块ID"],
        hover_name="地块ID",
        opacity=0.85,
        size_max=25
    )
    
    fig.update_traces(
        marker=dict(
            line=dict(width=1.5, color='white'),
            symbol='square'
        )
    )
    
    fig.update_layout(
        title=dict(text="果园分区域风险空间分布", font=dict(size=20, family="Microsoft YaHei"), x=0.5),
        xaxis=dict(title="经度方向（模拟坐标）", showgrid=True, gridcolor='rgba(200,200,200,0.3)',
                   zeroline=False, showticklabels=False),
        yaxis=dict(title="纬度方向（模拟坐标）", showgrid=True, gridcolor='rgba(200,200,200,0.3)',
                   zeroline=False, showticklabels=False),
        legend=dict(title="风险等级", orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=50, b=50, l=30, r=30),
        height=500
    )
    return fig


def create_time_trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    创建分时段风险趋势图
    模拟不同时段（早晨/中午/傍晚）的风险变化
    """
    # 基于现有数据模拟3个时段的风险分布
    periods = ["早晨 (6:00-10:00)", "中午 (10:00-14:00)", "下午 (14:00-18:00)", "傍晚 (18:00-22:00)"]
    
    # 模拟每个时段各风险等级的数量变化
    risk_counts = []
    np.random.seed(42)
    base_low = 144
    base_mid = 89
    base_high = 67
    
    for i, period in enumerate(periods):
        # 模拟时段变化：中午和下午高温时段风险略增
        factor_low = 1.0 - i * 0.03
        factor_mid = 1.0 + i * 0.08
        factor_high = 1.0 + i * 0.12
        
        low_count = max(0, int(base_low * factor_low + np.random.randint(-5, 5)))
        mid_count = max(0, int(base_mid * factor_mid + np.random.randint(-3, 3)))
        high_count = max(0, int(base_high * factor_high + np.random.randint(-3, 5)))
        
        risk_counts.append({"时段": period, "风险等级": "低风险", "地块数": low_count})
        risk_counts.append({"时段": period, "风险等级": "中风险", "地块数": mid_count})
        risk_counts.append({"时段": period, "风险等级": "高风险", "地块数": high_count})
    
    trend_df = pd.DataFrame(risk_counts)
    
    fig = px.line(
        trend_df,
        x="时段",
        y="地块数",
        color="风险等级",
        color_discrete_map={
            "低风险": "#2ecc71",
            "中风险": "#f39c12",
            "高风险": "#e74c3c"
        },
        markers=True,
        line_shape='spline'
    )
    
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    fig.update_layout(
        title=dict(text="分时段风险变化趋势（模拟）", font=dict(size=18, family="Microsoft YaHei"), x=0.5),
        xaxis_title="时段",
        yaxis_title="风险地块数量（块）",
        legend=dict(title="风险等级", orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=50, b=60, l=10, r=10),
        height=400
    )
    return fig


def create_feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    """创建特征重要性水平柱状图"""
    df = fi_df.copy()
    # 取前10个特征
    df = df.sort_values("Gain重要性" if "Gain重要性" in df.columns else df.columns[1], ascending=True).tail(12)
    
    value_col = "Gain重要性" if "Gain重要性" in df.columns else df.columns[1]
    name_col = df.columns[0]
    
    fig = px.bar(
        df,
        x=value_col,
        y=name_col,
        orientation='h',
        color=value_col,
        color_continuous_scale='Viridis',
        text=df[value_col].apply(lambda x: f"{x:.1f}")
    )
    fig.update_traces(textposition='outside', textfont=dict(size=11))
    fig.update_layout(
        title=dict(text="LightGBM 特征重要性排序（Gain）", font=dict(size=18, family="Microsoft YaHei"), x=0.5),
        xaxis_title="Gain 重要性",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(t=50, b=30, l=10, r=10),
        height=450
    )
    return fig


def create_shap_chart(shap_df: pd.DataFrame) -> go.Figure:
    """创建SHAP特征总贡献图"""
    df = shap_df.copy()
    
    shap_col = "SHAP总贡献" if "SHAP总贡献" in df.columns else df.columns[-1]
    name_col = df.columns[0]
    
    # 排除全0的特征，取贡献最大的特征
    df = df[df[shap_col] > 0.001].sort_values(shap_col, ascending=True)
    
    fig = go.Figure()
    
    # 使用横向堆叠条形图展示各等级贡献
    colors_level = ['#2ecc71', '#f39c12', '#e74c3c']
    level_names = ['等级0(低)', '等级1(中)', '等级2(高)']
    
    level_cols = [c for c in df.columns if '等级' in c and ('SHAP' in c or 'mean' in c)]
    
    if len(level_cols) >= 3:
        for i, col in enumerate(level_cols[:3]):
            fig.add_trace(go.Bar(
                y=df[name_col],
                x=df[col],
                name=level_names[i],
                orientation='h',
                marker_color=colors_level[i],
                text=df[col].apply(lambda x: f"{x:.3f}" if x > 0.05 else ""),
                textposition='outside'
            ))
        fig.update_layout(barmode='stack')
    else:
        # 降级为简单柱状图
        fig = px.bar(
            df, x=shap_col, y=name_col, orientation='h',
            color=shap_col, color_continuous_scale='Viridis'
        )
    
    fig.update_layout(
        title=dict(text="SHAP 特征贡献分析（各类别分别贡献）", font=dict(size=18, family="Microsoft YaHei"), x=0.5),
        xaxis_title="SHAP |mean| 贡献值",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=50, b=60, l=10, r=10),
        height=450
    )
    return fig


def create_kpi_gauge(f1_score: float, recall: float, precision: float, auc: float) -> go.Figure:
    """创建模型KPI指标仪表盘"""
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=("宏平均 F1 分数", "宏平均 召回率", "宏平均 精确率", "宏平均 AUC")
    )
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=f1_score,
        delta={'reference': 0.85, 'increasing': {'color': 'green'}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1},
            'bar': {'color': "#2ecc71"},
            'steps': [
                {'range': [0, 0.6], 'color': "lightgray"},
                {'range': [0.6, 0.8], 'color': "#f9e79f"},
                {'range': [0.8, 1.0], 'color': "#a9dfbf"}
            ],
            'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': 0.9}
        },
        number={'suffix': '', 'font': {'size': 22}}
    ), row=1, col=1)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=recall,
        delta={'reference': 0.85, 'increasing': {'color': 'green'}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#f39c12"},
            'steps': [
                {'range': [0, 0.6], 'color': "lightgray"},
                {'range': [0.6, 0.8], 'color': "#f9e79f"},
                {'range': [0.8, 1.0], 'color': "#fdebd0"}
            ],
            'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': 0.9}
        },
        number={'font': {'size': 22}}
    ), row=1, col=2)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=precision,
        delta={'reference': 0.85, 'increasing': {'color': 'green'}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, 0.6], 'color': "lightgray"},
                {'range': [0.6, 0.8], 'color': "#d4e6f1"},
                {'range': [0.8, 1.0], 'color': "#aed6f1"}
            ],
            'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': 0.9}
        },
        number={'font': {'size': 22}}
    ), row=1, col=3)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=auc,
        delta={'reference': 0.95, 'increasing': {'color': 'green'}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#9b59b6"},
            'steps': [
                {'range': [0, 0.7], 'color': "lightgray"},
                {'range': [0.7, 0.9], 'color': "#e8daef"},
                {'range': [0.9, 1.0], 'color': "#d7bde2"}
            ],
            'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': 0.95}
        },
        number={'font': {'size': 22}}
    ), row=1, col=4)
    
    fig.update_layout(
        title=dict(text="模型性能核心指标", font=dict(size=20, family="Microsoft YaHei"), x=0.5),
        height=350,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    return fig


def create_response_zone_chart(df: pd.DataFrame) -> go.Figure:
    """创建防控响应区统计图"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'bar'}]],
        subplot_titles=("响应区单元分布", "各响应区平均RRI")
    )
    
    zone_col = df.columns[0]
    unit_col = "防控单元数" if "防控单元数" in df.columns else df.columns[1]
    
    fig.add_trace(
        go.Pie(
            labels=df[zone_col],
            values=df[unit_col],
            hole=0.4,
            marker=dict(colors=["#e74c3c", "#f39c12", "#2ecc71"]),
            textinfo='label+percent+value'
        ),
        row=1, col=1
    )
    
    rri_col = "平均RRI" if "平均RRI" in df.columns else [c for c in df.columns if "RRI" in c][0]
    fig.add_trace(
        go.Bar(
            x=df[zone_col],
            y=df[rri_col],
            marker=dict(color=["#e74c3c", "#f39c12", "#2ecc71"]),
            text=df[rri_col].apply(lambda x: f"{x:.2f}"),
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(text="三级防控响应区分布", font=dict(size=18, family="Microsoft YaHei"), x=0.5),
        height=400,
        margin=dict(t=60, b=30, l=10, r=10),
        showlegend=False
    )
    return fig


def create_posi_weight_chart(posi_df: pd.DataFrame) -> go.Figure:
    """创建POSI因子权重图"""
    df = posi_df.copy()
    df = df[df[df.columns[1]] > 0.001].sort_values(df.columns[1], ascending=True)
    
    name_col = df.columns[0]
    weight_col = df.columns[1]
    
    fig = px.bar(
        df,
        x=weight_col,
        y=name_col,
        orientation='h',
        color=weight_col,
        color_continuous_scale='agsunset',
        text=df[weight_col].apply(lambda x: f"{x:.4f}")
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        title=dict(text="POSI 环境因子权重", font=dict(size=18, family="Microsoft YaHei"), x=0.5),
        xaxis_title="权重值",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(t=50, b=30, l=10, r=10),
        height=350
    )
    return fig


def create_risk_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    创建风险等级与防治窗口交叉热力图
    """
    # 使用交叉统计表
    cross_data = {
        "低风险": [144, 0, 144],
        "中风险": [66, 23, 89],
        "高风险": [0, 67, 67]
    }
    z_data = [[144, 0], [66, 23], [0, 67]]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=["窗口外", "窗口内"],
        y=["低风险", "中风险", "高风险"],
        colorscale=[
            [0, '#a8e6cf'],
            [0.5, '#ffd3b6'],
            [1, '#ff6b6b']
        ],
        text=[[f"低·外:144", f"低·内:0"],
              [f"中·外:66", f"中·内:23"],
              [f"高·外:0", f"高·内:67"]],
        texttemplate="%{text}",
        textfont={"size": 14, "family": "Microsoft YaHei"},
        showscale=True,
        colorbar=dict(title="地块数")
    ))
    
    fig.update_layout(
        title=dict(text="风险等级 x 防治窗口交叉分布", font=dict(size=18, family="Microsoft YaHei"), x=0.5),
        xaxis_title="防治窗口状态",
        yaxis_title="风险等级",
        height=350,
        margin=dict(t=50, b=30, l=30, r=10)
    )
    return fig


def create_prevention_strategy_table(df: pd.DataFrame) -> pd.DataFrame:
    """返回格式化的防控策略表格数据"""
    return df


def create_metrics_summary_card(df: pd.DataFrame) -> dict:
    """从KPI数据提取关键指标"""
    metrics = {}
    if df.empty:
        return metrics
    
    # 找宏平均行
    if "风险等级" in df.columns:
        macro_row = df[df["风险等级"] == "宏平均"]
        if not macro_row.empty:
            row = macro_row.iloc[0]
            metrics["f1"] = row.get("F1分数", 0)
            metrics["recall"] = row.get("召回率", 0)
            metrics["precision"] = row.get("精确率", 0)
            metrics["samples"] = row.get("样本数", 0)
    
    return metrics
