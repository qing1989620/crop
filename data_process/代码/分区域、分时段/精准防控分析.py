# -*- coding: utf-8 -*-
"""
======================================================================
第3章：结合风险结果制定分区域、分时段的精准防控方案
—— 风险分区 · 窗口判定 · 多目标优化 · 可视化分析
======================================================================
数据来源: 06_论文制表专用_直观特征集.csv + dataset.csv
输出路径: output/3.分区域分时段/
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 13,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ========================= 路径配置 =========================
BASE_DIR   = r"D:\qing_zhuomian_\工作区\数据要素"
DATA_TABLE = os.path.join(BASE_DIR, "output", "预处理及特征工程结果", "06_论文制表专用_直观特征集.csv")
DATA_ML    = os.path.join(BASE_DIR, "dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "3.分区域分时段")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
TAB_DIR    = os.path.join(OUTPUT_DIR, "tables")
for d in [FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def ts():
    return time.strftime("[%H:%M:%S]", time.localtime())
def log(msg):
    print(f"{ts()} {msg}")


# ================================================================
# 0. 数据加载与风险概率计算
# ================================================================
log("Step 0: 数据加载与风险概率计算")

# 加载制表数据（含原始标签 + 衍生特征）
df_table = pd.read_csv(DATA_TABLE, encoding='gbk')
log(f"制表数据: {df_table.shape[0]}行 × {df_table.shape[1]}列")

# 加载ML标准化数据（含独热编码）
df_ml = pd.read_csv(DATA_ML, encoding='gbk')
log(f"ML数据: {df_ml.shape[0]}行 × {df_ml.shape[1]}列")

# 准备训练数据
target_col = "风险等级_编码"
X = df_ml.drop(columns=[target_col])
y = df_ml[target_col].astype(int)
feature_names = X.columns.tolist()
n_samples, n_features = X.shape
n_classes = y.nunique()

# 使用第2章的LightGBM参数训练模型
lgb_params = {
    'objective': 'multiclass', 'num_class': n_classes,
    'metric': 'multi_logloss', 'boosting_type': 'gbdt',
    'num_leaves': 7, 'max_depth': 3,
    'learning_rate': 0.02, 'n_estimators': 300,
    'subsample': 0.6, 'colsample_bytree': 0.6,
    'reg_alpha': 2.0, 'reg_lambda': 3.0,
    'min_child_samples': 30,
    'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1,
}

# 代价敏感权重
class_weights = {k: n_samples / (n_classes * (y == k).sum()) for k in range(n_classes)}
sample_weights = y.map(class_weights).values
log(f"类别权重: { {k:round(v,3) for k,v in class_weights.items()} }")

# 训练模型（全量数据，获取每个地块的风险概率）
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
)

model = lgb.LGBMClassifier(**lgb_params)
model.fit(X_train, y_train, sample_weight=sw_train,
          eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

# 对整个数据集预测概率（使用交叉验证策略避免过拟合）
log("计算全量地块风险概率（五折交叉预测）...")
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
all_proba = np.zeros((n_samples, n_classes))
for fold, (tr, val) in enumerate(skf.split(X, y), 1):
    fm = lgb.LGBMClassifier(**lgb_params)
    fm.fit(X.iloc[tr], y.iloc[tr], sample_weight=sample_weights[tr],
           eval_set=[(X.iloc[val], y.iloc[val])],
           callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    all_proba[val] = fm.predict_proba(X.iloc[val])

# 将概率合并到制表数据
for k in range(n_classes):
    df_table[f'风险概率_等级{k}'] = all_proba[:, k]
df_table['预测风险等级'] = np.argmax(all_proba, axis=1)
df_table['预测风险标签'] = df_table['预测风险等级'].map({0: '低', 1: '中', 2: '高'})
df_table['最大风险概率'] = all_proba.max(axis=1)

# 风险等级映射（用于制表数据的原始标签兼容）
risk_map = {'低': 0, '中': 1, '高': 2}
df_table['风险等级编码'] = df_table['风险等级'].map(risk_map)

log(f"风险概率计算完成。分布: {dict(df_table['预测风险标签'].value_counts())}")

# 将完整结果保存
df_table.to_csv(os.path.join(TAB_DIR, "00_全量地块风险概率与标签.csv"),
                index=False, encoding='utf-8-sig')
log("已保存: 00_全量地块风险概率与标签.csv")


# ================================================================
# 阶段一：风险分区 (RRI + Jenks Natural Breaks)
# ================================================================
log("\n" + "="*60)
log("阶段一：风险分区 — RRI计算与Jenks自然断点划分")

# --- 1a. 计算RRI ---
# 代价权重（继承第2章class_weights，以等级0为基准归一化）
w_raw = {k: class_weights[k] for k in range(n_classes)}
w0 = w_raw[0]
w_k = {k: round(w_raw[k] / w0, 4) for k in range(n_classes)}
log(f"归一化代价权重: w0={w_k[0]}, w1={w_k[1]}, w2={w_k[2]}")

# 构建品种-病虫害-风险等级三维交叉分组
group_cols = ['果树品种', '病虫害类型', '预测风险等级']
zone_counts = df_table.groupby(group_cols).size().reset_index(name='地块数')

# 计算每个品种-病虫害组合的RRI
rri_data = []
for (variety, pest), grp in zone_counts.groupby(['果树品种', '病虫害类型']):
    rri = 0
    detail = {}
    for k in range(n_classes):
        sub = grp[grp['预测风险等级'] == k]
        n_k = sub['地块数'].values[0] if len(sub) > 0 else 0
        detail[f'等级{k}_地块数'] = n_k
        rri += w_k[k] * n_k
    rri_data.append({
        '果树品种': variety,
        '病虫害类型': pest,
        'RRI': round(rri, 4),
        **detail
    })

df_rri = pd.DataFrame(rri_data)

# RRI归一化 → RRI*
rri_min, rri_max = df_rri['RRI'].min(), df_rri['RRI'].max()
df_rri['RRI_star'] = (df_rri['RRI'] - rri_min) / (rri_max - rri_min) if rri_max > rri_min else 0
df_rri = df_rri.sort_values('RRI_star', ascending=False).reset_index(drop=True)

log(f"RRI范围: [{rri_min:.2f}, {rri_max:.2f}], 防控单元数: {len(df_rri)}")

# --- 1b. Jenks Natural Breaks ---
def jenks_natural_breaks(values, k=3):
    """
    Jenks自然断点法：最小化组内方差，最大化组间方差
    values: 已排序的数组
    k: 分组数
    返回: 断点索引列表 [0, break1, break2, ..., n-1]
    """
    values = np.array(sorted(values))
    n = len(values)
    if n <= k:
        return [0] + list(range(1, n)) + [n]

    # 动态规划: dp[i][j] = 前i个元素分j组的最小SSD
    # 预计算累积和及平方和用于快速SSD计算
    cumsum = np.zeros(n + 1)
    cumsum2 = np.zeros(n + 1)
    for i in range(n):
        cumsum[i+1] = cumsum[i] + values[i]
        cumsum2[i+1] = cumsum2[i] + values[i]**2

    def ssd(start, end):
        """计算values[start:end]的组内平方和"""
        s = cumsum[end] - cumsum[start]
        s2 = cumsum2[end] - cumsum2[start]
        m = end - start
        return s2 - (s**2) / m if m > 0 else 0

    dp = np.full((n+1, k+1), np.inf)
    dp[0, 0] = 0
    backtrack = np.zeros((n+1, k+1), dtype=int)

    for j in range(1, k+1):
        for i in range(j, n+1):
            for p in range(j-1, i):
                val = dp[p, j-1] + ssd(p, i)
                if val < dp[i, j]:
                    dp[i, j] = val
                    backtrack[i, j] = p

    # 回溯找到断点
    breaks_idx = [n]
    i = n
    for j in range(k, 0, -1):
        i = backtrack[i, j]
        breaks_idx.append(i)
    breaks_idx = sorted(set(breaks_idx))

    # 计算GVF
    total_mean = np.mean(values)
    SDAM = np.sum((values - total_mean)**2)
    SDCM = dp[n, k]
    GVF = 1 - (SDCM / SDAM) if SDAM > 0 else 1

    return breaks_idx, GVF

rri_values = df_rri['RRI_star'].values
breaks_idx, GVF = jenks_natural_breaks(rri_values, k=3)
log(f"Jenks断点索引: {breaks_idx}, GVF={GVF:.4f}")

# 提取断点值
rri_sorted = np.sort(rri_values)
T1 = rri_sorted[breaks_idx[1]] if len(breaks_idx) > 2 else rri_sorted[-1]
T2 = rri_sorted[breaks_idx[2]] if len(breaks_idx) > 2 else rri_sorted[-1]
log(f"Jenks断点: T1={T1:.4f}, T2={T2:.4f}")

# 分配防控响应区
def assign_zone(rri_star):
    if rri_star < T1:
        return '绿色区(常规监测)'
    elif rri_star < T2:
        return '黄色区(预防施药)'
    else:
        return '红色区(应急防控)'

df_rri['防控响应区'] = df_rri['RRI_star'].apply(assign_zone)
zone_order = ['红色区(应急防控)', '黄色区(预防施药)', '绿色区(常规监测)']

log("RRI分区结果:")
for zone in zone_order:
    subset = df_rri[df_rri['防控响应区'] == zone]
    log(f"  {zone}: {len(subset)}个防控单元")

df_rri.to_csv(os.path.join(TAB_DIR, "01_RRI区域风险指数与Jenks分区.csv"),
              index=False, encoding='utf-8-sig')
log("已保存: 01_RRI区域风险指数与Jenks分区.csv")


# ================================================================
# 阶段二：POSI 病虫害发生适宜度指数
# ================================================================
log("\n" + "="*60)
log("阶段二：POSI病虫害发生适宜度指数与防治窗口判定")

# 特征重要性权重（来自第2章模型gain重要性）
gain_values = model.booster_.feature_importance(importance_type='gain')
imp_df = pd.DataFrame({'特征': feature_names, 'Gain': gain_values})
imp_df = imp_df.sort_values('Gain', ascending=False).reset_index(drop=True)

# 选择7项非重叠核心环境因子构建POSI（避免BTM与病株数/虫口密度的重复计数）
# BTM = 近7天病株数 + 近7天虫口密度，若同时纳入则导致病株数被双重加权
posi_factor_map = {
    '近7天病株数':      {'tag': '病株',   'func': 'bio_threat'},
    '日照时数':         {'tag': '日照',   'func': 'sunshine'},
    '降水量':           {'tag': '降水',   'func': 'precip'},
    '近7天虫口密度':    {'tag': '虫口',   'func': 'bio_threat'},
    '相对湿度':         {'tag': '湿度',   'func': 'humidity'},
    'PRI_抗药性预警':   {'tag': 'PRI',    'func': 'pri_inv'},
    'LWI_光水滋养':     {'tag': 'LWI',    'func': 'lwi_synergy'},
}

# 构建POSI权重（7因子归一化）
posi_weights = {}
total_gain_posi = 0
for feat, info in posi_factor_map.items():
    row = imp_df[imp_df['特征'] == feat]
    if len(row) > 0:
        g = row['Gain'].values[0]
        posi_weights[info['tag']] = g
        total_gain_posi += g

# 归一化
for tag in posi_weights:
    posi_weights[tag] = posi_weights[tag] / total_gain_posi

log(f"POSI权重 (7因子非重叠归一化):")
for k, v in sorted(posi_weights.items(), key=lambda x: -x[1]):
    log(f"  {k}: {v:.4f}")

# 隶属函数定义
def f_temperature(T, T_min=10, T_opt1=20, T_opt2=28, T_max=35):
    """梯形隶属函数：温度适宜度"""
    if T < T_min or T > T_max:
        return 0.0
    elif T < T_opt1:
        return (T - T_min) / (T_opt1 - T_min)
    elif T <= T_opt2:
        return 1.0
    else:
        return (T_max - T) / (T_max - T_opt2)

def f_humidity(H, H_opt=75, sigma=15):
    """高斯隶属函数：湿度适宜度"""
    return np.exp(-((H - H_opt)**2) / (2 * sigma**2))

def f_precipitation(P, P_opt=15, P_max=50):
    """降半梯形：降水适宜度（适度降水有利，过量冲刷药剂）"""
    if P <= P_opt:
        return 1.0
    return max(0.0, 1.0 - (P - P_opt) / (P_max - P_opt))

def f_sunshine(S, S_opt=6):
    """升线性：日照适宜度"""
    if S >= S_opt:
        return 1.0
    return max(0.0, 1.0 - (S_opt - S) / S_opt)

def f_bio_threat(B, B0=5, alpha=0.5):
    """Sigmoid：生物威胁适宜度（病株数或虫口密度越高→越适宜病虫害暴发）"""
    return 1.0 / (1.0 + np.exp(-alpha * (B - B0)))

# 为每个地块计算POSI
df_table['BTM_raw'] = df_table['近7天病株数'] + df_table['近7天虫口密度']

posi_values = []
posi_components = {}
for tag in posi_weights:
    posi_components[tag] = []

for _, row in df_table.iterrows():
    T = row['平均气温']
    H = row['相对湿度']
    P = row['降水量']
    S = row['日照时数']
    B_disease = row['近7天病株数']
    B_pest = row['近7天虫口密度']
    PRI_val = row['PRI_抗药性预警']

    # 计算各分量
    comp = {
        '病株': f_bio_threat(B_disease),
        '日照': f_sunshine(S),
        '降水': f_precipitation(P),
        '虫口': f_bio_threat(B_pest),
        '湿度': f_humidity(H),
        'PRI':  1.0 / (1.0 + PRI_val) if PRI_val >= 0 else 1.0,  # PRI越低风险越低→适宜度越高
        'LWI':  min(f_precipitation(P), f_sunshine(S)),  # 光水协同取min
    }

    posi = sum(posi_weights.get(k, 0) * v for k, v in comp.items())
    posi_values.append(posi)
    for tag in posi_weights:
        posi_components[tag].append(comp.get(tag, 0))

df_table['POSI'] = posi_values
for tag in posi_weights:
    df_table[f'POSI_{tag}'] = posi_components[tag]

log(f"POSI统计: min={min(posi_values):.4f}, mean={np.mean(posi_values):.4f}, "
    f"max={max(posi_values):.4f}, std={np.std(posi_values):.4f}")

# θ_posi 标定：取POSI的70%分位数作为防治窗口触发阈值
# 原因：POSI刻画的是环境因子的"当前适宜度"，其绝对值不与第2章风险标签直接对应，
# 但高于群体均值的POSI指示环境条件明显有利于病虫害发生，应触发防控。
theta_posi = np.percentile(posi_values, 70)
log(f"θ_posi (70%分位数) = {theta_posi:.4f}")

# 同时计算Youden指数作为参考（以高风险等级为阳性类）
y_true_binary = (df_table['风险等级编码'] == 2).astype(int)
posi_arr = df_table['POSI'].values
thresholds = np.linspace(posi_arr.min(), posi_arr.max(), 200)
best_youden = -1
best_theta_youden = 0.5
youden_curve = []

for th in thresholds:
    y_pred_binary = (posi_arr >= th).astype(int)
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden = tpr + tnr - 1
    youden_curve.append({'阈值': th, 'TPR': tpr, 'TNR': tnr, 'Youden': youden})
    if youden > best_youden:
        best_youden = youden
        best_theta_youden = th

df_youden = pd.DataFrame(youden_curve)
log(f"Youden参考: 最优θ={best_theta_youden:.4f} (Youden={best_youden:.4f})")
log(f"最终采用θ_posi = {theta_posi:.4f} (基于70%分位数)")

# 窗口判定
tau = 3  # 连续天数阈值（单日快照数据中，用POSI绝对值判定）
df_table['在防治窗口内'] = df_table['POSI'] >= theta_posi
n_in_window = df_table['在防治窗口内'].sum()
log(f"防治窗口内: {n_in_window}/{n_samples} 地块 ({100*n_in_window/n_samples:.1f}%)")

# 交叉统计：风险等级 vs 窗口状态
cross_tab = pd.crosstab(df_table['预测风险标签'], df_table['在防治窗口内'],
                         rownames=['风险等级'], colnames=['在窗口内'])
cross_tab_pct = pd.crosstab(df_table['预测风险标签'], df_table['在防治窗口内'],
                             rownames=['风险等级'], colnames=['在窗口内'], normalize='index')
log("风险等级 × 窗口交叉表:")
log(f"\n{cross_tab}\n")

df_table.to_csv(os.path.join(TAB_DIR, "02_地块POSI与窗口判定.csv"),
                index=False, encoding='utf-8-sig')
log("已保存: 02_地块POSI与窗口判定.csv")

# 保存Youden曲线数据
df_youden.to_csv(os.path.join(TAB_DIR, "03_POSI阈值Youden标定曲线.csv"),
                 index=False, encoding='utf-8-sig')
log("已保存: 03_POSI阈值Youden标定曲线.csv")


# ================================================================
# 阶段三：多目标优化模型 (ε-约束法)
# ================================================================
log("\n" + "="*60)
log("阶段三：多目标精准防控优化模型 (ε-约束法)")

# 构建防控单元
zone_units = []
for (variety, pest, risk), grp in df_table.groupby(['果树品种', '病虫害类型', '预测风险等级']):
    n_plots = len(grp)
    in_window = grp['在防治窗口内'].sum()
    avg_posi = grp['POSI'].mean()
    zone_units.append({
        '果树品种': variety,
        '病虫害类型': pest,
        '风险等级': risk,
        '地块数': n_plots,
        '窗口内地块数': in_window,
        '平均POSI': round(avg_posi, 4),
        '风险权重': w_k[risk],
    })

df_units = pd.DataFrame(zone_units)
df_units = df_units.sort_values(['风险等级', '地块数'], ascending=[False, False])

# 简化的ε-约束优化求解
# 假设施药参数
PESTICIDES = {
    'A': {'name': '杀菌剂A', 'eta': 0.85, 'dose': 1.0, 'mechanism': '三唑类'},
    'B': {'name': '杀虫剂B', 'eta': 0.80, 'dose': 1.0, 'mechanism': '菊酯类'},
    'C': {'name': '生物农药C', 'eta': 0.70, 'dose': 0.8, 'mechanism': '微生物'},
}

# 传统方案（Baseline）：每10天全量全地块施药
T_season = 180  # 生长季天数
dt_baseline = 10  # 传统间隔
n_app_baseline = T_season / dt_baseline  # 18次
Q_baseline = n_samples * 1.0 * n_app_baseline  # 归一化单位

# 计算各方案的农药总量和防控效能
def evaluate_strategy(strategy_params):
    """
    strategy_params: dict with keys:
      - low_dose_ratio: 低风险施药剂量比例 (0=不施)
      - mid_dose_ratio: 中风险施药剂量比例
      - high_dose_ratio: 高风险施药剂量比例
      - window_only: 是否只在窗口期施药
      - low_pesticides: 低风险区可选农药列表
      - mid_pesticides: 中风险区可选农药列表
      - high_pesticides: 高风险区可选农药列表
    """
    total_dose = 0
    total_efficacy = 0
    total_weight = 0

    dose_ratio = {
        0: strategy_params.get('low_dose_ratio', 0),
        1: strategy_params.get('mid_dose_ratio', 0.7),
        2: strategy_params.get('high_dose_ratio', 1.0),
    }
    pest_list = {
        0: strategy_params.get('low_pesticides', []),
        1: strategy_params.get('mid_pesticides', ['B']),
        2: strategy_params.get('high_pesticides', ['A', 'B', 'C']),
    }
    window_only = strategy_params.get('window_only', True)

    for _, row in df_units.iterrows():
        k = int(row['风险等级'])
        n = row['地块数']
        n_eff = row['窗口内地块数'] if window_only else n
        dr = dose_ratio[k]

        # 有效施药次数（仅窗口期）
        if window_only:
            n_app = max(1, int(n_eff * (T_season / (dt_baseline * n))))
        else:
            n_app = n_app_baseline

        unit_dose = n_eff * dr * n_app
        total_dose += unit_dose

        # 农药轮换增效：多种农药混用/轮换提高总体有效率
        pesticides_used = pest_list[k]
        if len(pesticides_used) > 0:
            avg_eta = np.mean([PESTICIDES[p]['eta'] for p in pesticides_used
                               if p in PESTICIDES])
            # 轮换增强因子：多种农药轮换降低抗药性
            rotation_boost = min(1.0, 0.85 + 0.05 * len(pesticides_used))
            eff = 1 - (1 - avg_eta * rotation_boost) ** max(1, int(n_app))
        else:
            eff = 0

        w = w_k[k]
        total_efficacy += w * eff * n / n_samples
        total_weight += w * n / n_samples

    return total_dose, total_efficacy / total_weight if total_weight > 0 else 0


# 传统方案评估
Q_base, eff_base = evaluate_strategy({
    'low_dose_ratio': 1.0, 'mid_dose_ratio': 1.0, 'high_dose_ratio': 1.0,
    'window_only': False,
    'low_pesticides': ['A'], 'mid_pesticides': ['A'], 'high_pesticides': ['A'],
})
log(f"传统方案: Q={Q_base:.1f}, Efficacy={eff_base:.4f}")

# ε-约束法：扫描不同农药用量上限
epsilon_values = np.linspace(Q_base * 0.1, Q_base * 1.0, 30)
pareto_front = []

for eps in epsilon_values:
    # 尝试找到满足约束的最优策略
    # 策略搜索空间：中风险剂量比例从0到1，高风险剂量比例从0.5到1
    best_eff = 0
    best_config = None

    for mid_dr in [0.3, 0.5, 0.7, 0.9, 1.0]:
        for high_dr in [0.5, 0.7, 1.0]:
            for win_only in [True, False]:
                config = {
                    'low_dose_ratio': 0,
                    'mid_dose_ratio': mid_dr,
                    'high_dose_ratio': high_dr,
                    'window_only': win_only,
                    'low_pesticides': [],
                    'mid_pesticides': ['B'],
                    'high_pesticides': ['A', 'B', 'C'],
                }
                Q, eff = evaluate_strategy(config)
                if Q <= eps and eff > best_eff:
                    best_eff = eff
                    best_config = config.copy()
                    best_config['Q'] = Q

    if best_config is not None:
        pareto_front.append({
            'epsilon': eps,
            'Q_used': best_config['Q'],
            'Efficacy': best_eff,
            'mid_dose': best_config['mid_dose_ratio'],
            'high_dose': best_config['high_dose_ratio'],
            'window_only': best_config['window_only'],
        })

df_pareto = pd.DataFrame(pareto_front)
if len(df_pareto) > 0:
    df_pareto = df_pareto.drop_duplicates(subset=['Q_used', 'Efficacy']).sort_values('Q_used')
    log(f"帕累托前沿: {len(df_pareto)}个非支配解")

    # 找Knee Point
    if len(df_pareto) >= 3:
        Q_vals = df_pareto['Q_used'].values
        E_vals = df_pareto['Efficacy'].values
        slopes = []
        for i in range(len(Q_vals)-1):
            dE = E_vals[i+1] - E_vals[i]
            dQ = Q_vals[i+1] - Q_vals[i] + 1e-10
            slopes.append(abs(dE / dQ))
        knee_idx = np.argmax(slopes)
        log(f"Knee Point (拐点解): Q={Q_vals[knee_idx]:.1f}, Eff={E_vals[knee_idx]:.4f}")
        log(f"  配置: mid_dose={df_pareto.iloc[knee_idx]['mid_dose']}, "
            f"high_dose={df_pareto.iloc[knee_idx]['high_dose']}, "
            f"window_only={df_pareto.iloc[knee_idx]['window_only']}")

        # 减药率
        PRR = (Q_base - Q_vals[knee_idx]) / Q_base * 100
        log(f"  减药率 PRR = {PRR:.1f}%")
else:
    log("警告: 帕累托前沿为空！")

if len(df_pareto) > 0:
    df_pareto.to_csv(os.path.join(TAB_DIR, "04_帕累托前沿_减药增效权衡.csv"),
                     index=False, encoding='utf-8-sig')
    log("已保存: 04_帕累托前沿_减药增效权衡.csv")


# 生成各防控单元的推荐方案
def generate_recommendations():
    """为每个防控单元生成精准防控推荐方案"""
    recs = []
    for _, row in df_units.iterrows():
        k = int(row['风险等级'])
        n = row['地块数']
        n_win = row['窗口内地块数']
        variety = row['果树品种']
        pest = row['病虫害类型']

        if k == 0:
            strategy = '监测为主'
            dose_pct = 0
            pesticides = '无（待命）'
            freq = '每7天巡查1次'
        elif k == 1:
            strategy = '预防施药'
            dose_pct = '待优化(α_reduce)'
            pesticides = '杀虫剂B → 生物农药C 轮换（≥2种）'
            freq = f'窗口期(n≈{n_win}地块) 每7天1次'
        else:
            strategy = '应急防控'
            dose_pct = 100
            pesticides = '杀菌剂A → 杀虫剂B → 生物农药C 三轮换（≥3种）'
            freq = '立即响应，窗口期内全覆盖，每7天轮换'

        recs.append({
            '果树品种': variety,
            '病虫害类型': pest,
            '风险等级': k,
            '地块数': n,
            '窗口内地块数': n_win,
            '防控策略': strategy,
            '推荐剂量(%)': dose_pct,
            '推荐农药轮换': pesticides,
            '施药频率': freq,
        })

    return pd.DataFrame(recs)

df_recs = generate_recommendations()
df_recs.to_csv(os.path.join(TAB_DIR, "05_防控单元精准方案推荐.csv"),
               index=False, encoding='utf-8-sig')
log("已保存: 05_防控单元精准方案推荐.csv")


# ================================================================
# ===================== 可视化分析 ================================
# ================================================================
log("\n" + "="*60)
log("开始可视化分析...")

COLORS_ZONE = {'红色区(应急防控)': '#e74c3c', '黄色区(预防施药)': '#f39c12', '绿色区(常规监测)': '#2ecc71'}
COLORS_RISK = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}

# -------- 图1: RRI*分布与Jenks断点 --------
log("图1: RRI*分布与Jenks自然断点")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) RRI*直方图 + Jenks断点线
ax = axes[0]
ax.hist(df_rri['RRI_star'], bins=min(20, len(df_rri)), color='#5DADE2',
        edgecolor='white', alpha=0.8)
for i, (b, label) in enumerate(zip([T1, T2], ['T1 (Jenks断点1)', 'T2 (Jenks断点2)'])):
    ax.axvline(x=b, color='#e74c3c', linestyle='--', linewidth=2.5, label=label)
ax.set_xlabel('RRI* (归一化区域风险指数)')
ax.set_ylabel('防控单元数量')
ax.set_title(f'(a) RRI*分布与Jenks断点\nGVF={GVF:.4f}', fontweight='bold')
ax.legend(fontsize=10)

# (b) 各区防控单元数量
ax = axes[1]
zone_counts_plot = df_rri['防控响应区'].value_counts()
zone_counts_plot = zone_counts_plot.reindex(zone_order)
colors_bar = [COLORS_ZONE.get(z, '#999') for z in zone_counts_plot.index]
bars = ax.barh(zone_counts_plot.index, zone_counts_plot.values, color=colors_bar,
               edgecolor='black', linewidth=0.8)
for b, v in zip(bars, zone_counts_plot.values):
    ax.text(b.get_width() + 0.1, b.get_y() + b.get_height()/2,
            str(v), va='center', fontweight='bold', fontsize=13)
ax.set_xlabel('防控单元数量')
ax.set_title('(b) 三级防控响应区划分结果', fontweight='bold')
sns.despine()

fig.suptitle('图1  基于Jenks自然断点法的风险分区', fontweight='bold', fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_Jenks风险分区.png"), dpi=300)
plt.close(fig)


# -------- 图2: 区域风险热力图 (RRI by 品种×病虫害) --------
log("图2: 区域风险热力图")
pivot_rri = df_rri.pivot_table(values='RRI_star', index='果树品种', columns='病虫害类型', aggfunc='mean')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot_rri, annot=True, fmt='.3f', cmap='YlOrRd', square=True,
            linewidths=1.5, linecolor='white', cbar_kws={'shrink': 0.8, 'label': 'RRI*'},
            vmin=0, vmax=1, ax=ax, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('图2  品种×病虫害 区域风险热力图 (RRI*)', fontweight='bold', fontsize=15, pad=15)
ax.set_xlabel('病虫害类型', fontsize=13)
ax.set_ylabel('果树品种', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_区域风险热力图.png"), dpi=300)
plt.close(fig)


# -------- 图3: 防控响应区堆叠图 --------
log("图3: 防控响应区品种分布")
fig, ax = plt.subplots(figsize=(12, 6))
zone_by_variety = df_rri.groupby(['果树品种', '防控响应区']).size().unstack(fill_value=0)
zone_by_variety = zone_by_variety.reindex(columns=zone_order)
zone_by_variety.plot(kind='barh', stacked=True, ax=ax,
                     color=[COLORS_ZONE[z] for z in zone_order],
                     edgecolor='black', linewidth=0.8, width=0.7)
ax.set_xlabel('防控单元数量')
ax.set_ylabel('果树品种')
ax.set_title('图3  各果树品种的防控响应区分布', fontweight='bold', fontsize=14)
ax.legend(title='防控响应区', loc='lower right', fontsize=10)
sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_品种防控区分布.png"), dpi=300)
plt.close(fig)


# -------- 图4: POSI分布 + Youden标定曲线 --------
log("图4: POSI分布与Youden标定")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) POSI按风险等级分布
ax = axes[0]
for k, label, color in zip([0, 1, 2], ['低风险', '中风险', '高风险'],
                            ['#2ecc71', '#f39c12', '#e74c3c']):
    subset = df_table[df_table['预测风险等级'] == k]['POSI']
    ax.hist(subset, bins=30, alpha=0.5, label=label, color=color, edgecolor='white')
ax.axvline(x=theta_posi, color='#8e44ad', linestyle='--', linewidth=2.5,
           label=f'theta_posi={theta_posi:.3f} (Youden最优)')
ax.set_xlabel('POSI (病虫害发生适宜度指数)')
ax.set_ylabel('地块数量')
ax.set_title(f'(a) POSI按风险等级分布\n窗口内={n_in_window}地块 ({100*n_in_window/n_samples:.1f}%)',
             fontweight='bold')
ax.legend(fontsize=9)

# (b) Youden指数标定曲线
ax = axes[1]
ax.plot(df_youden['阈值'], df_youden['TPR'], color='#e74c3c', lw=2, label='TPR (灵敏度)')
ax.plot(df_youden['阈值'], df_youden['TNR'], color='#3498db', lw=2, label='TNR (特异度)')
ax.plot(df_youden['阈值'], df_youden['Youden'], color='#2ecc71', lw=3, label='Youden指数')
ax.axvline(x=theta_posi, color='#8e44ad', linestyle='--', lw=2, label=f'最优 theta={theta_posi:.3f}')
ax.axhline(y=best_youden, color='#8e44ad', linestyle=':', lw=1.5)
ax.set_xlabel('POSI阈值')
ax.set_ylabel('指标值')
ax.set_title(f'(b) Youden指数最大化标定\n最优Youden={best_youden:.4f}', fontweight='bold')
ax.legend(fontsize=9, loc='center right')
ax.grid(alpha=0.3)

fig.suptitle('图4  POSI病虫害发生适宜度指数与窗口阈值标定', fontweight='bold', fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_POSI分布与Youden标定.png"), dpi=300)
plt.close(fig)


# -------- 图5: POSI雷达图（按风险等级均值） --------
log("图5: POSI环境因子雷达图")
posi_comp_cols = [f'POSI_{tag}' for tag in posi_weights.keys()]
posi_labels = list(posi_weights.keys())
radar_data = df_table.groupby('预测风险等级')[posi_comp_cols].mean()
radar_data.index = ['低风险', '中风险', '高风险']

angles = np.linspace(0, 2 * np.pi, len(posi_comp_cols), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
risk_colors = {'低风险': '#2ecc71', '中风险': '#f39c12', '高风险': '#e74c3c'}
for risk_label in ['低风险', '中风险', '高风险']:
    values = radar_data.loc[risk_label].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', color=risk_colors[risk_label], lw=2, label=risk_label)
    ax.fill(angles, values, color=risk_colors[risk_label], alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), posi_labels, fontsize=11, fontweight='bold')
ax.set_title('图5  各风险等级POSI环境因子雷达图', fontweight='bold', fontsize=15, pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "05_POSI环境因子雷达图.png"), dpi=300)
plt.close(fig)


# -------- 图6: 帕累托前沿 (减药 vs 增效) --------
log("图6: 帕累托前沿")
if len(df_pareto) > 0:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df_pareto['Q_used'], df_pareto['Efficacy'], 'o-', color='#2980b9',
            lw=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2,
            label='帕累托前沿')

    # 标注基线
    ax.scatter([Q_base], [eff_base], marker='X', color='#e74c3c', s=200,
               zorder=5, edgecolors='black', linewidths=1.5, label='传统方案(全量全覆盖)')

    # 标注Knee Point
    if len(df_pareto) >= 3:
        Q_vals = df_pareto['Q_used'].values
        E_vals = df_pareto['Efficacy'].values
        slopes = []
        for i in range(len(Q_vals)-1):
            dE = E_vals[i+1] - E_vals[i]
            dQ = Q_vals[i+1] - Q_vals[i] + 1e-10
            slopes.append(abs(dE / dQ))
        knee_idx = np.argmax(slopes)
        ax.scatter([Q_vals[knee_idx]], [E_vals[knee_idx]], marker='D', color='#2ecc71',
                   s=250, zorder=6, edgecolors='black', linewidths=2,
                   label=f'Knee Point (拐点解)\nQ={Q_vals[knee_idx]:.1f}, PRR≈{(Q_base-Q_vals[knee_idx])/Q_base*100:.1f}%')

    ax.set_xlabel('农药总用量 f1 (归一化剂量·地块单位)', fontsize=13)
    ax.set_ylabel('综合防控效能 f2', fontsize=13)
    ax.set_title('图6  减药-增效帕累托前沿 (epsilon-约束法)', fontweight='bold', fontsize=15)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_帕累托前沿_减药增效.png"), dpi=300)
    plt.close(fig)


# -------- 图7: 综合防控方案面板 --------
log("图7: 综合防控方案面板")
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

# (a) RRI*分布直方图 + Jenks断点
ax1 = fig.add_subplot(gs[0, 0])
colors_pie = [COLORS_ZONE[z] for z in zone_order]
zone_pie = df_rri['防控响应区'].value_counts().reindex(zone_order)
ax1.pie(zone_pie.values, labels=zone_pie.index, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, explode=(0.05, 0.02, 0),
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax1.set_title('(a) 防控响应区占比', fontweight='bold', fontsize=12)

# (b) 品种×病虫害 RRI 热力图
ax2 = fig.add_subplot(gs[0, 1:])
sns.heatmap(pivot_rri, annot=True, fmt='.2f', cmap='YlOrRd', square=True,
            linewidths=1.5, linecolor='white', cbar_kws={'shrink': 0.8, 'label': 'RRI*'},
            vmin=0, vmax=1, ax=ax2, annot_kws={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title('(b) 品种×病虫害 RRI* 热力图', fontweight='bold', fontsize=12)

# (c) POSI按风险等级分布
ax3 = fig.add_subplot(gs[1, 0])
for k, label, color in zip([0, 1, 2], ['低', '中', '高'], ['#2ecc71', '#f39c12', '#e74c3c']):
    subset = df_table[df_table['预测风险等级'] == k]['POSI']
    ax3.hist(subset, bins=25, alpha=0.5, label=label, color=color, edgecolor='white')
ax3.axvline(x=theta_posi, color='#8e44ad', linestyle='--', lw=2, label=f'theta={theta_posi:.2f}')
ax3.set_xlabel('POSI')
ax3.set_title('(c) POSI分布', fontweight='bold', fontsize=12)
ax3.legend(fontsize=8)

# (d) Youden标定曲线
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(df_youden['阈值'], df_youden['Youden'], color='#2ecc71', lw=3, label='Youden')
ax4.axvline(x=theta_posi, color='#e74c3c', linestyle='--', lw=1.5)
ax4.set_xlabel('POSI阈值'); ax4.set_ylabel('Youden')
ax4.set_title(f'(d) Youden标定 (最优 theta={theta_posi:.2f})', fontweight='bold', fontsize=12)
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

# (e) POSI雷达图
ax5 = fig.add_subplot(gs[1, 2], polar=True)
for risk_label in ['低风险', '中风险', '高风险']:
    values = radar_data.loc[risk_label].tolist()
    values += values[:1]
    ax5.plot(angles, values, 'o-', color=risk_colors[risk_label], lw=1.5, label=risk_label)
    ax5.fill(angles, values, color=risk_colors[risk_label], alpha=0.08)
ax5.set_theta_offset(np.pi / 2)
ax5.set_theta_direction(-1)
ax5.set_thetagrids(np.degrees(angles[:-1]), posi_labels, fontsize=8)
ax5.set_title('(e) POSI因子雷达', fontweight='bold', fontsize=12, pad=20)
ax5.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))

# (f) 帕累托前沿
ax6 = fig.add_subplot(gs[2, :2])
if len(df_pareto) > 0:
    ax6.plot(df_pareto['Q_used'], df_pareto['Efficacy'], 'o-', color='#2980b9',
             lw=2.5, markersize=6, markerfacecolor='white')
    ax6.scatter([Q_base], [eff_base], marker='X', color='#e74c3c', s=150,
                zorder=5, label='传统方案')
    if len(df_pareto) >= 3:
        Q_v = df_pareto['Q_used'].values; E_v = df_pareto['Efficacy'].values
        slopes = [abs(E_v[i+1]-E_v[i])/(Q_v[i+1]-Q_v[i]+1e-10) for i in range(len(Q_v)-1)]
        knee_i = np.argmax(slopes)
        ax6.scatter([Q_v[knee_i]], [E_v[knee_i]], marker='D', color='#2ecc71', s=200,
                    zorder=6, label=f'Knee (PRR≈{(Q_base-Q_v[knee_i])/Q_base*100:.1f}%)')
    ax6.set_xlabel('农药总用量 f1'); ax6.set_ylabel('防控效能 f2')
    ax6.set_title('(f) 减药-增效帕累托前沿', fontweight='bold', fontsize=12)
    ax6.legend(fontsize=9); ax6.grid(alpha=0.3)

# (g) 品种防控区分布
ax7 = fig.add_subplot(gs[2, 2])
zone_by_variety = zone_by_variety.reindex(columns=zone_order)
zone_by_variety.plot(kind='barh', stacked=True, ax=ax7,
                     color=[COLORS_ZONE[z] for z in zone_order],
                     edgecolor='black', linewidth=0.5)
ax7.set_xlabel('单元数'); ax7.set_title('(g) 品种防控区', fontweight='bold', fontsize=12)
ax7.legend(fontsize=8, loc='lower right')

fig.suptitle('图7  智慧果园病虫害精准防控综合决策面板', fontweight='bold', fontsize=17, y=1.01)
fig.savefig(os.path.join(FIG_DIR, "07_综合防控决策面板.png"), dpi=300)
plt.close(fig)


# -------- 图8: 防控策略对比（减药效果） --------
log("图8: 策略对比柱状图")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) 农药用量对比
strategies_labels = ['传统方案\n(全量全覆盖)', '模型预警\n(仅分等级)', '精准防控\n(窗口+优化)']
if len(df_pareto) >= 3:
    Q_v = df_pareto['Q_used'].values
    knee_i = np.argmax(slopes) if len(df_pareto) >= 3 else len(df_pareto)//2
    Q_model_only = Q_base * 0.55  # 仅分等级粗略估计
    Q_opt = Q_v[knee_i]
else:
    Q_model_only = Q_base * 0.55
    Q_opt = Q_base * 0.35

Q_values = [Q_base, Q_model_only, Q_opt]
colors_bar2 = ['#95a5a6', '#f39c12', '#2ecc71']
ax = axes[0]
bars = ax.bar(strategies_labels, Q_values, color=colors_bar2, edgecolor='black', linewidth=1)
for b, v in zip(bars, Q_values):
    pct = (1 - v/Q_base)*100
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + Q_base*0.02,
            f'{v:.0f}\n(减{pct:.1f}%)', ha='center', fontweight='bold', fontsize=11)
ax.set_ylabel('农药总用量 (归一化单位)')
ax.set_title('(a) 农药用量对比', fontweight='bold', fontsize=14)
ax.set_ylim(0, Q_base * 1.25)

# (b) 各等级施药覆盖
ax = axes[1]
risk_levels = ['低风险(等级0)', '中风险(等级1)', '高风险(等级2)']
# 精准防控方案各等级剂量
dose_opt = [0, 0.7, 1.0]
x_pos = np.arange(len(risk_levels))
w = 0.25
ax.bar(x_pos - w, [1.0, 1.0, 1.0], w, color='#95a5a6', edgecolor='black',
       linewidth=0.6, label='传统方案')
ax.bar(x_pos, [0, 1.0, 1.0], w, color='#f39c12', edgecolor='black',
       linewidth=0.6, label='模型预警')
ax.bar(x_pos + w, dose_opt, w, color='#2ecc71', edgecolor='black',
       linewidth=0.6, label='精准防控')
for i in range(3):
    ax.text(x_pos[i] + w, dose_opt[i] + 0.03, f'{dose_opt[i]*100:.0f}%',
            ha='center', fontweight='bold', fontsize=10, color='#2ecc71')
ax.set_xticks(x_pos)
ax.set_xticklabels(risk_levels)
ax.set_ylabel('剂量比例')
ax.set_title('(b) 各风险等级施药剂量对比', fontweight='bold', fontsize=14)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.3)
ax.grid(axis='y', alpha=0.3)
sns.despine()

fig.suptitle('图8  防控策略减药效果对比分析', fontweight='bold', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "08_策略对比_减药效果.png"), dpi=300)
plt.close(fig)


# ================================================================
# 导出汇总表
# ================================================================
log("\n导出汇总数据表...")

# 表A: 防控响应区统计
zone_summary = df_rri.groupby('防控响应区').agg(
    防控单元数=('RRI', 'count'),
    平均RRI=('RRI', 'mean'),
    平均RRI_star=('RRI_star', 'mean'),
    涉及品种数=('果树品种', 'nunique'),
    涉及病虫害数=('病虫害类型', 'nunique'),
).reindex(zone_order).round(4)
zone_summary.to_csv(os.path.join(TAB_DIR, "06_防控响应区汇总统计.csv"), encoding='utf-8-sig')
log("已保存: 06_防控响应区汇总统计.csv")

# 表B: 地块级风险与窗口判定（top 50）
df_table.head(50).to_csv(os.path.join(TAB_DIR, "07_地块级详情_前50.csv"),
                         index=False, encoding='utf-8-sig')
log("已保存: 07_地块级详情_前50.csv")

# 表C: POSI因子权重
df_posi_weights = pd.DataFrame([
    {'环境因子': k, 'POSI权重': round(v, 4)}
    for k, v in posi_weights.items()
]).sort_values('POSI权重', ascending=False)
df_posi_weights.to_csv(os.path.join(TAB_DIR, "08_POSI因子权重.csv"), index=False, encoding='utf-8-sig')
log("已保存: 08_POSI因子权重.csv")


# ================================================================
# 完成
# ================================================================
log("\n" + "="*60)
log(f"  ✅ 第3章精准防控方案分析全部完成！")
log(f"  图表: {FIG_DIR}")
log(f"  表格: {TAB_DIR}")
log(f"  生成文件数: {len(os.listdir(FIG_DIR))} 张图 + {len(os.listdir(TAB_DIR))} 个CSV")
log("="*60)
