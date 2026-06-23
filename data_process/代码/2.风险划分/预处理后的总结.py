# -*- coding: utf-8 -*-
"""
对预处理后的 dataset.csv 进行统计描述，生成 log.txt 到 output/2.低中高/ 下
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ===== 路径配置 =====
DATA_PATH = r"D:\qing_zhuomian_\工作区\数据要素\dataset.csv"
OUTPUT_DIR = r"D:\qing_zhuomian_\工作区\数据要素\output\2.低中高"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "log.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 读取数据 =====
# 尝试多种编码读取（Windows 中文环境常见 GBK 编码）
for enc in ['gbk', 'gb2312', 'gb18030', 'utf-8', 'latin-1']:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        print(f"使用编码 {enc} 读取成功")
        break
    except (UnicodeDecodeError, Exception):
        continue
else:
    raise RuntimeError("无法识别文件编码，请检查 dataset.csv")
print(f"数据读取完成，形状: {df.shape}")

# ===== 写入 log =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write(f"  预处理数据集统计报告 — dataset.csv\n")
    f.write(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

    # ---------- 1. 基本信息 ----------
    f.write("【1】数据集基本信息\n")
    f.write("-" * 40 + "\n")
    f.write(f"  样本数 (行): {df.shape[0]}\n")
    f.write(f"  特征数 (列): {df.shape[1]}\n")
    f.write(f"  内存占用: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n")
    f.write(f"  列名列表:\n")
    for i, col in enumerate(df.columns, 1):
        f.write(f"    {i:2d}. {col}\n")
    f.write("\n")

    # ---------- 2. 缺失值统计 ----------
    f.write("【2】缺失值统计\n")
    f.write("-" * 40 + "\n")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            f.write(f"  {col}: 缺失 {missing[col]} 个 ({missing_pct[col]:.2f}%)\n")
            has_missing = True
    if not has_missing:
        f.write("  ✓ 无缺失值\n")
    f.write("\n")

    # ---------- 3. 数据类型 ----------
    f.write("【3】数据类型分布\n")
    f.write("-" * 40 + "\n")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        f.write(f"  {dtype}: {count} 列\n")
    f.write("\n")

    # ---------- 4. 数值型特征描述统计 ----------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    f.write(f"【4】数值型特征描述统计 (共 {len(numeric_cols)} 列)\n")
    f.write("-" * 40 + "\n")

    if numeric_cols:
        desc = df[numeric_cols].describe().T
        # 添加额外的统计量
        desc['median'] = df[numeric_cols].median()
        desc['skew'] = df[numeric_cols].skew()
        desc['kurtosis'] = df[numeric_cols].kurtosis()
        desc['range'] = desc['max'] - desc['min']
        desc['cv'] = desc['std'] / desc['mean'].abs()  # 变异系数

        for idx, row in desc.iterrows():
            f.write(f"\n  ▶ {idx}\n")
            f.write(f"    样本数(count): {row['count']:.0f}\n")
            f.write(f"    均值(mean):    {row['mean']:.6f}\n")
            f.write(f"    标准差(std):   {row['std']:.6f}\n")
            f.write(f"    最小值(min):   {row['min']:.6f}\n")
            f.write(f"    25%分位(Q1):   {row['25%']:.6f}\n")
            f.write(f"    中位数(median):{row['median']:.6f}\n")
            f.write(f"    75%分位(Q3):   {row['75%']:.6f}\n")
            f.write(f"    最大值(max):   {row['max']:.6f}\n")
            f.write(f"    极差(range):   {row['range']:.6f}\n")
            f.write(f"    偏度(skew):    {row['skew']:.6f}\n")
            f.write(f"    峰度(kurtosis):{row['kurtosis']:.6f}\n")
            f.write(f"    变异系数(CV):  {row['cv']:.6f}\n")
    f.write("\n")

    # ---------- 5. 二值/独热编码列统计 ----------
    binary_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0, 1})]
    if binary_cols:
        f.write(f"【5】二值型特征分布 (品种/病虫害独热编码, 共 {len(binary_cols)} 列)\n")
        f.write("-" * 40 + "\n")
        for col in binary_cols:
            counts = df[col].value_counts().sort_index()
            f.write(f"  {col}:\n")
            for val, cnt in counts.items():
                f.write(f"    值={int(val)}: {cnt} 个 ({cnt/len(df)*100:.2f}%)\n")
        f.write("\n")

    # ---------- 6. 目标变量统计（风险等级） ----------
    target_col = "风险等级_风险"
    if target_col in df.columns:
        f.write(f"【6】目标变量「{target_col}」分布\n")
        f.write("-" * 40 + "\n")
        target_counts = df[target_col].value_counts().sort_index()
        for val, cnt in target_counts.items():
            f.write(f"  类别 {int(val)}: {cnt} 个 ({cnt/len(df)*100:.2f}%)\n")
        f.write("\n")

    # ---------- 7. 相关系数摘要 ----------
    if len(numeric_cols) >= 2:
        f.write(f"【7】数值型特征相关系数矩阵（绝对值 TOP-20）\n")
        f.write("-" * 40 + "\n")
        corr_matrix = df[numeric_cols].corr().abs()
        # 取上三角
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_pairs = (
            upper.stack()
            .sort_values(ascending=False)
        )
        top_n = min(20, len(corr_pairs))
        for rank, ((col_a, col_b), corr_val) in enumerate(
            corr_pairs.head(top_n).items(), 1
        ):
            f.write(f"  {rank:2d}. {col_a} ↔ {col_b}: r = {corr_val:.4f}\n")
        f.write("\n")

    # ---------- 8. 按目标变量分组统计 ----------
    if target_col in df.columns and numeric_cols:
        f.write(f"【8】按「{target_col}」分组的数值特征均值对比\n")
        f.write("-" * 40 + "\n")
        group_means = df.groupby(target_col)[numeric_cols].mean().T
        group_means.columns = [f"类别{int(c)}" for c in group_means.columns]
        for feat_name, row in group_means.iterrows():
            f.write(f"  {feat_name}:\n")
            for grp, val in row.items():
                f.write(f"    {grp}: {val:.6f}\n")
        f.write("\n")

    # ---------- 9. 重复行检测 ----------
    f.write("【9】重复行检测\n")
    f.write("-" * 40 + "\n")
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        f.write(f"  存在 {dup_count} 行完全重复 ({dup_count/len(df)*100:.2f}%)\n")
    else:
        f.write(f"  ✓ 无完全重复行\n")
    f.write("\n")

    # ---------- 10. 数据摘要 ----------
    f.write("【10】综合分析摘要\n")
    f.write("-" * 40 + "\n")
    f.write(f"  • 数据集包含 {df.shape[0]} 条样本，{df.shape[1]} 个特征\n")
    f.write(f"  • 数值型特征 {len(numeric_cols)} 个\n")
    f.write(f"  • 二值型特征 {len(binary_cols)} 个\n")
    if target_col in df.columns:
        n_classes = df[target_col].nunique()
        f.write(f"  • 目标变量「{target_col}」共 {n_classes} 个类别\n")
        dominant = df[target_col].value_counts().index[0]
        dominant_pct = df[target_col].value_counts().iloc[0] / len(df) * 100
        f.write(f"    多数类为 类别{int(dominant)}，占比 {dominant_pct:.2f}%\n")
    f.write(f"  • 无缺失值，数据完整性良好\n" if not has_missing else f"  • 存在缺失值，需进一步处理\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("  报告结束\n")
    f.write("=" * 70 + "\n")

print(f"统计报告已生成 → {OUTPUT_FILE}")
