# -*- coding: utf-8 -*-
"""
=============================================================================
第3章：病虫害风险预警模型的构建与多分类决策机理
—— LightGBM 代价敏感多分类预警引擎
=============================================================================
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, log_loss,
)
import lightgbm as lgb

import importlib.util
HAS_SHAP = importlib.util.find_spec("shap") is not None
if HAS_SHAP:
    import shap  # type: ignore

# ==================== 全局设置（学习预处理.py 的正确写法）====================
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')  # ← 关键：seaborn 直接绑定字体

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 13,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

DATA_PATH  = r"D:\qing_zhuomian_\工作区\数据要素\dataset.csv"
OUTPUT_DIR = r"D:\qing_zhuomian_\工作区\数据要素\output\2.低中高"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
TAB_DIR    = os.path.join(OUTPUT_DIR, "tables")
for d in [FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
COLORS_CLS = ['#2ecc71', '#f39c12', '#e74c3c']

def ts():
    return time.strftime("[%H:%M:%S]", time.localtime())
def log(msg):
    print(f"{ts()} {msg}")


# ================================================================
# 1. 数据加载
# ================================================================
log("Step 1: 数据加载")
for enc in ['gbk', 'gb2312', 'gb18030', 'utf-8']:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        break
    except Exception:
        continue

target_col = "风险等级_编码"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)
LEAKY = ['病虫害_0', '病虫害_1', '病虫害_2', '病虫害_3']
X = X.drop(columns=[c for c in LEAKY if c in X.columns], errors='ignore')

feature_names = X.columns.tolist()
n_samples, n_features = X.shape
n_classes = y.nunique()
class_labels = sorted(y.unique().tolist())
class_names = [f'低风险(等级{k})' if k==0 else f'中风险(等级{k})' if k==1 else f'高风险(等级{k})' for k in class_labels]

log(f"样本={n_samples}  特征={n_features}  类别={n_classes}")
log(f"分布: {dict(y.value_counts().sort_index())}")


# ================================================================
# 2. 代价敏感权重 + 划分
# ================================================================
log("Step 2: 代价敏感权重")
class_weights = {k: n_samples / (n_classes * (y == k).sum()) for k in class_labels}
sample_weights = y.map(class_weights).values
log(f"权重: { {k:round(v,3) for k,v in class_weights.items()} }")

X_train, X_test, y_train, y_test, sw_train, _ = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
)
log(f"训练={X_train.shape[0]}  测试={X_test.shape[0]}")


# ================================================================
# 3. 模型 & 五折 CV（强化正则化，压制过拟合）
# ================================================================
lgb_params = {
    'objective': 'multiclass', 'num_class': n_classes,
    'metric': 'multi_logloss', 'boosting_type': 'gbdt',
    'num_leaves': 7,               # ↓ 极简树结构
    'max_depth': 3,                # ↓ 浅层树
    'learning_rate': 0.02,         # ↓ 慢学
    'n_estimators': 300,           # ↓ 限制迭代
    'subsample': 0.6,              # ↓ 强行采样
    'colsample_bytree': 0.6,       # ↓ 强列采样
    'reg_alpha': 2.0,              # ↑↑ L1 强正则
    'reg_lambda': 3.0,             # ↑↑ L2 强正则 Ω(θ)
    'min_child_samples': 30,       # ↑ 大叶子最小样本
    'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1,
}
model = lgb.LGBMClassifier(**lgb_params)

log("Step 3: 五折 CV（同时收集 OOF 概率用于真实 AUC）")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = {'acc': [], 'macro_f1': [], 'log_loss': []}
oof_proba = np.zeros((n_samples, n_classes))  # Out-of-Fold 概率
oof_true = np.zeros(n_samples)

for fold, (tr, val) in enumerate(skf.split(X, y), 1):
    fm = lgb.LGBMClassifier(**lgb_params)
    fm.fit(X.iloc[tr], y.iloc[tr], sample_weight=sample_weights[tr],
           eval_set=[(X.iloc[val], y.iloc[val])],
           callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    yp = fm.predict(X.iloc[val])
    ypb = fm.predict_proba(X.iloc[val])
    cv_scores['acc'].append(accuracy_score(y.iloc[val], yp))
    cv_scores['macro_f1'].append(f1_score(y.iloc[val], yp, average='macro'))
    cv_scores['log_loss'].append(log_loss(y.iloc[val], ypb))
    oof_proba[val] = ypb
    oof_true[val] = y.iloc[val].values
    log(f"  Fold{fold} Acc={cv_scores['acc'][-1]:.4f} F1={cv_scores['macro_f1'][-1]:.4f} Loss={cv_scores['log_loss'][-1]:.4f}")
    log(f"  Fold{fold} Acc={cv_scores['acc'][-1]:.4f} F1={cv_scores['macro_f1'][-1]:.4f} Loss={cv_scores['log_loss'][-1]:.4f}")

log(f"  CV: Acc={np.mean(cv_scores['acc']):.4f}±{np.std(cv_scores['acc']):.4f}  "
    f"Macro-F1={np.mean(cv_scores['macro_f1']):.4f}±{np.std(cv_scores['macro_f1']):.4f}")

pd.DataFrame({'Fold':range(1,6),'Accuracy':cv_scores['acc'],'Macro_F1':cv_scores['macro_f1'],'Log_Loss':cv_scores['log_loss']}) \
    .to_csv(os.path.join(TAB_DIR, "CV_五折交叉验证.csv"), index=False, encoding='utf-8-sig')


# ================================================================
# 4. 最终训练 & 测试
# ================================================================
log("Step 4: 最终训练")
model.fit(X_train, y_train, sample_weight=sw_train,
          eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=class_labels)

per_class_precision = precision_score(y_test, y_pred, average=None)
per_class_recall    = recall_score(y_test, y_pred, average=None)
per_class_f1        = f1_score(y_test, y_pred, average=None)

log("\n========== 测试集报告 ==========")
print(classification_report(y_test, y_pred, target_names=[f'等级{k}' for k in class_labels]))
log(f"高风险召回率(核心KPI) = {per_class_recall[-1]:.4f}")
log(f"Macro-F1              = {np.mean(per_class_f1):.4f}")

pd.DataFrame(classification_report(y_test, y_pred,
    target_names=[f'等级{k}' for k in class_labels], output_dict=True)).T \
    .to_csv(os.path.join(TAB_DIR, "测试集分类报告.csv"), encoding='utf-8-sig')


# ================================================================
# 可视化 1 — 类别分布
# ================================================================
log("图1: 类别分布")
fig, ax = plt.subplots(figsize=(5, 4))
counts = y.value_counts().sort_index()
bars = ax.bar(class_names, counts.values, color=COLORS_CLS, edgecolor='black', linewidth=0.8)
for b, c in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+2, f'{c}\n({c/n_samples*100:.1f}%)',
            ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('样本数量'); ax.set_title('图1  风险等级类别分布', fontweight='bold')
sns.despine(); fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_类别分布.png")); plt.close(fig)


# ================================================================
# 可视化 2 — 混淆矩阵
# ================================================================
log("图2: 混淆矩阵")
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
labels_2d = [f'等级{k}' for k in class_labels]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=labels_2d, yticklabels=labels_2d,
            ax=axes[0], linewidths=1, linecolor='white', cbar_kws={'shrink':0.8})
axes[0].set_title('(a) 混淆矩阵（样本数）', fontweight='bold')
axes[0].set_xlabel('预测标签'); axes[0].set_ylabel('真实标签')

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd', square=True,
            xticklabels=labels_2d, yticklabels=labels_2d,
            ax=axes[1], linewidths=1, linecolor='white', vmin=0, vmax=1, cbar_kws={'shrink':0.8})
axes[1].set_title('(b) 混淆矩阵（行归一化）', fontweight='bold')
axes[1].set_xlabel('预测标签'); axes[1].set_ylabel('真实标签')
fig.suptitle('图2  测试集混淆矩阵', fontweight='bold', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_混淆矩阵.png")); plt.close(fig)

pd.DataFrame(cm, index=[f'真实_等级{k}' for k in class_labels],
             columns=[f'预测_等级{k}' for k in class_labels]) \
    .to_csv(os.path.join(TAB_DIR, "混淆矩阵.csv"), encoding='utf-8-sig')


# ================================================================
# 可视化 3 — ROC 曲线（基于 OOF 跨折叠预测，非单次测试集，更诚实）
# ================================================================
log("图3: ROC 曲线 (基于 OOF 五折跨折叠预测)")
fig, ax = plt.subplots(figsize=(7, 6.5))
oof_true_bin = label_binarize(oof_true.astype(int), classes=class_labels)
macro_auc = 0
for i, k in enumerate(class_labels):
    if oof_true_bin[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(oof_true_bin[:, i], oof_proba[:, i])
    roc_auc = auc(fpr, tpr)
    macro_auc += roc_auc
    ax.plot(fpr, tpr, color=COLORS_CLS[i], lw=2.5, label=f'等级{k} (AUC={roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS_CLS[i])
macro_auc /= n_classes
ax.plot([0,1], [0,1], 'k--', lw=1.2, alpha=0.6, label='随机分类器')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('假阳性率 (FPR)'); ax.set_ylabel('真阳性率 (TPR)')
ax.set_title(f'图3  ROC 曲线 (五折OOF, One-vs-Rest)\nMacro-AUC = {macro_auc:.4f}', fontweight='bold')
ax.legend(loc='lower right', frameon=True, fontsize=10); ax.grid(alpha=0.3)
sns.despine(); fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_ROC曲线.png")); plt.close(fig)

pd.DataFrame({'风险等级': [f'等级{k}' for k in class_labels] + ['宏平均'],
              'AUC_OOF': [auc(*roc_curve(oof_true_bin[:,i], oof_proba[:,i])[:2])
                          for i in range(n_classes)] + [macro_auc]}) \
    .to_csv(os.path.join(TAB_DIR, "ROC_AUC值.csv"), index=False, encoding='utf-8-sig')


# ================================================================
# 可视化 4 — 特征重要性
# ================================================================
log("图4: 特征重要性")
imp_df = pd.DataFrame({
    '特征': feature_names,
    'Gain重要性': model.booster_.feature_importance(importance_type='gain'),
    'Split次数': model.booster_.feature_importance(importance_type='split'),
}).sort_values('Gain重要性', ascending=False).reset_index(drop=True)
imp_df.to_csv(os.path.join(TAB_DIR, "特征重要性.csv"), index=False, encoding='utf-8-sig')

top_n = min(15, len(imp_df))
top = imp_df.head(top_n).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top['特征'], top['Gain重要性'],
        color=plt.cm.viridis(np.linspace(0.15, 0.9, top_n)),
        edgecolor='gray', linewidth=0.5)
for i, (_, row) in enumerate(top.iterrows()):
    ax.text(row['Gain重要性']+max(top['Gain重要性'])*0.01, i,
            f'{row["Gain重要性"]:.1f}', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Gain 重要性'); ax.set_title(f'图4  LightGBM 特征重要性 Top{top_n}', fontweight='bold')
sns.despine(); fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_特征重要性.png")); plt.close(fig)


# ================================================================
# 可视化 5 — SHAP 分析
# ================================================================
if HAS_SHAP:
    log("图5: SHAP 可解释性")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_sample = X_test.iloc[:min(100, len(X_test))].copy()
    shap_raw = explainer.shap_values(shap_sample)

    if isinstance(shap_raw, list):
        shap_list = [np.array(sv) for sv in shap_raw]
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        shap_list = [shap_raw[:,:,i] for i in range(shap_raw.shape[2])]
    else:
        shap_list = [shap_raw]

    for i in range(min(len(shap_list), n_classes)):
        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_list[i], shap_sample, feature_names=feature_names,
                          show=False, max_display=15)
        fig.suptitle(f'图5-{chr(97+i)}  SHAP 特征影响 — 风险等级 {class_labels[i]}',
                     fontweight='bold', fontsize=14, y=0.98)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"05_SHAP_等级{class_labels[i]}.png"))
        plt.close('all')
    log("  SHAP summary 完成")

    try:
        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_list, shap_sample, feature_names=feature_names,
                          plot_type='bar', class_names=[f'等级{k}' for k in class_labels[:len(shap_list)]],
                          show=False, max_display=15)
        fig.suptitle('图6  SHAP 全局特征重要性', fontweight='bold', fontsize=14, y=0.98)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "06_SHAP_全局条形图.png"))
        plt.close('all')
        log("  SHAP 全局图完成")
    except Exception as e:
        log(f"  SHAP 全局图跳过: {e}")

    shap_abs = np.array([np.abs(sv).mean(axis=0) for sv in shap_list]).T
    shap_df = pd.DataFrame(shap_abs, columns=[f'SHAP|mean|_等级{k}' for k in class_labels[:shap_abs.shape[1]]])
    shap_df.insert(0, '特征', feature_names)
    shap_df['SHAP总贡献'] = shap_df.iloc[:,1:].sum(axis=1)
    shap_df.sort_values('SHAP总贡献', ascending=False) \
        .to_csv(os.path.join(TAB_DIR, "SHAP特征贡献.csv"), index=False, encoding='utf-8-sig')
else:
    log("图5: SHAP 跳过（未安装）")


# ================================================================
# 可视化 6 — 分类性能对比（核心 KPI）
# ================================================================
log("图7: 分类性能对比（核心 KPI）")
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(n_classes); w = 0.25
ax.bar(x-w, per_class_precision, w, color='#3498db', edgecolor='black', linewidth=0.6, label='精确率')
ax.bar(x,   per_class_recall,    w, color='#e74c3c', edgecolor='black', linewidth=0.6, label='召回率')
ax.bar(x+w, per_class_f1,        w, color='#2ecc71', edgecolor='black', linewidth=0.6, label='F1 分数')
for i in range(n_classes):
    ax.text(x[i]-w, per_class_precision[i]+0.02, f'{per_class_precision[i]:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(x[i],   per_class_recall[i]+0.02,    f'{per_class_recall[i]:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(x[i]+w, per_class_f1[i]+0.02,        f'{per_class_f1[i]:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'等级{k}\n(n={int((y_test==k).sum())})' for k in class_labels])
ax.set_ylabel('分数')
ax.set_title(f'图7  各风险等级分类性能对比\n【核心KPI】高风险召回率 = {per_class_recall[-1]:.4f}', fontweight='bold')
ax.legend(loc='upper right', frameon=True, fontsize=10, ncol=3)
ax.set_ylim(0, 1.15); ax.grid(axis='y', alpha=0.3)
sns.despine(); fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "07_分类性能对比.png")); plt.close(fig)

pd.DataFrame({
    '风险等级': [f'等级{k}' for k in class_labels] + ['宏平均'],
    '精确率': list(per_class_precision) + [np.mean(per_class_precision)],
    '召回率': list(per_class_recall) + [np.mean(per_class_recall)],
    'F1分数': list(per_class_f1) + [np.mean(per_class_f1)],
    '样本数': [int((y_test==k).sum()) for k in class_labels] + [len(y_test)],
}).to_csv(os.path.join(TAB_DIR, "核心KPI指标.csv"), index=False, encoding='utf-8-sig')


# ================================================================
# 可视化 7 — 综合性能面板
# ================================================================
log("图8: 综合性能面板")
fig = plt.figure(figsize=(18, 13))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.pie(counts.values, labels=[f'等级{k}' for k in class_labels],
        colors=COLORS_CLS, autopct='%1.1f%%', startangle=90,
        explode=(0, 0.02, 0.05), wedgeprops={'edgecolor':'white','linewidth':1.5})
ax1.set_title('类别分布', fontweight='bold', fontsize=12)

ax2 = fig.add_subplot(gs[0, 1])
bp = ax2.boxplot([cv_scores['acc'], cv_scores['macro_f1'], cv_scores['log_loss']],
                 labels=['Accuracy','Macro-F1','LogLoss'], patch_artist=True, widths=0.5)
for p, c in zip(bp['boxes'], ['#3498db','#e74c3c','#f39c12']):
    p.set_facecolor(c); p.set_alpha(0.6)
ax2.set_title('五折 CV 分布', fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd', square=True,
            xticklabels=labels_2d, yticklabels=labels_2d,
            ax=ax3, linewidths=1, linecolor='white', cbar_kws={'shrink':0.7})
ax3.set_title('混淆矩阵（归一化）', fontweight='bold', fontsize=12)

ax4 = fig.add_subplot(gs[1, :2])
for i, k in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(oof_true_bin[:,i], oof_proba[:,i])
    ax4.plot(fpr, tpr, color=COLORS_CLS[i], lw=2, label=f'等级{k} (AUC={auc(fpr,tpr):.3f})')
ax4.plot([0,1],[0,1],'k--',lw=1); ax4.set_xlabel('FPR'); ax4.set_ylabel('TPR')
ax4.set_title(f'ROC 曲线(OOF)  Macro-AUC={macro_auc:.3f}', fontweight='bold', fontsize=12)
ax4.legend(fontsize=9, loc='lower right'); ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[1, 2])
top10 = imp_df.head(10).iloc[::-1]
ax5.barh(top10['特征'], top10['Gain重要性'],
         color=plt.cm.plasma(np.linspace(0.1,0.9,10)), edgecolor='gray', linewidth=0.5)
ax5.set_xlabel('Gain 重要性'); ax5.set_title('Top-10 特征', fontweight='bold', fontsize=12)

ax6 = fig.add_subplot(gs[2, :])
x6 = np.arange(n_classes); w6 = 0.25
ax6.bar(x6-w6, per_class_precision, w6, color='#3498db', label='Precision')
ax6.bar(x6,    per_class_recall,    w6, color='#e74c3c', label='Recall')
ax6.bar(x6+w6, per_class_f1,        w6, color='#2ecc71', label='F1')
for i in range(n_classes):
    ax6.text(x6[i]-w6, per_class_precision[i]+0.02, f'{per_class_precision[i]:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax6.text(x6[i],    per_class_recall[i]+0.02,    f'{per_class_recall[i]:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax6.text(x6[i]+w6, per_class_f1[i]+0.02,        f'{per_class_f1[i]:.3f}', ha='center', fontsize=8, fontweight='bold')
ax6.set_xticks(x6); ax6.set_xticklabels(labels_2d); ax6.set_ylim(0, 1.15); ax6.set_ylabel('分数')
ax6.set_title(f'各类别性能  |  高风险召回率={per_class_recall[-1]:.4f}  |  Macro-F1={np.mean(per_class_f1):.4f}',
              fontweight='bold', fontsize=13)
ax6.legend(ncol=3, fontsize=10, loc='upper right'); ax6.grid(axis='y', alpha=0.3)

fig.suptitle('图8  病虫害风险预警模型综合性能评估面板', fontweight='bold', fontsize=17, y=1.01)
fig.savefig(os.path.join(FIG_DIR, "08_综合性能面板.png")); plt.close(fig)


# ================================================================
# 论文制表
# ================================================================
pd.DataFrame({'参数':list(lgb_params.keys()), '取值':[str(v) for v in lgb_params.values()]}) \
    .to_csv(os.path.join(TAB_DIR, "模型超参数.csv"), index=False, encoding='utf-8-sig')
pd.DataFrame({'风险等级':[f'{k}' for k in class_labels],
              '样本数':counts.values, '占比(%)':(counts.values/n_samples*100).round(2)}) \
    .to_csv(os.path.join(TAB_DIR, "类别分布.csv"), index=False, encoding='utf-8-sig')

log("\n"+"="*50)
log(f"  ✅ 全部完成！图表 {FIG_DIR}  |  表格 {TAB_DIR}")
log(f"  核心KPI: 高风险召回率={per_class_recall[-1]:.4f}  Macro-F1={np.mean(per_class_f1):.4f}")
log("="*50)
