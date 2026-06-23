# -*- coding: utf-8 -*-
"""
=============================================================================
统一数据处理管线 — 果园病虫害风险预警
=============================================================================
输入: 原始数据集 CSV（如 data_process/data.csv）
输出: output/ 目录下的全部模型产出（CSV + 图表）

三阶段流水线:
  阶段一: 预处理与特征工程 → output/1.预处理及特征工程结果/
  阶段二: LightGBM 风险建模 + SHAP 可解释性 → output/2.低中高/
  阶段三: 精准防控方案 (POSI/RRI/Jenks/帕累托) → output/3.分区域分时段/

用法:
  python pipeline.py --input data_process/data.csv
  python pipeline.py --input /path/to/your_data.csv
=============================================================================
"""

import os, sys, warnings, time, argparse, traceback
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ==================== 路径工具 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.makedirs(os.path.join(OUTPUT_DIR, "1.预处理及特征工程结果"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "2.低中高", "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "2.低中高", "tables"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "3.分区域分时段", "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "3.分区域分时段", "tables"), exist_ok=True)

COLORS_CLS = ['#2ecc71', '#f39c12', '#e74c3c']

def ts():
    return time.strftime("[%H:%M:%S]", time.localtime())

def log(msg):
    print(f"{ts()} {msg}")


# ===================================================================
#  阶段一: 预处理与特征工程
# ===================================================================
class Stage1_Preprocessing:
    """数据预处理与特征工程，输出制表用数据和ML标准化矩阵"""

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.out_dir = os.path.join(OUTPUT_DIR, "1.预处理及特征工程结果")
        os.makedirs(self.out_dir, exist_ok=True)
        self.df = None
        self.df_table = None
        self.df_ml = None

    def run(self):
        log("=" * 60)
        log("阶段一：预处理与特征工程")
        self._load()
        self._feature_engineering()
        self._export()
        log("Stage 1 completed")
        return self

    def _load(self):
        log(f"读取: {self.input_path}")
        for enc in ['gbk', 'gb2312', 'gb18030', 'utf-8-sig', 'utf-8']:
            try:
                self.df = pd.read_csv(self.input_path, encoding=enc)
                log(f"  编码={enc}, 形状={self.df.shape}")
                return
            except (UnicodeDecodeError, Exception):
                continue
        raise RuntimeError(f"无法读取文件: {self.input_path}")

    def _feature_engineering(self):
        log("特征派生中...")
        df = self.df.copy()

        # 高阶衍生特征
        df['THI_温湿胁迫'] = df['平均气温'] * df['相对湿度']
        df['BTM_生物威胁动量'] = df['近7天病株数'] + df['近7天虫口密度']
        df['PRI_抗药性预警'] = df['BTM_生物威胁动量'] / (df['近30天用药次数'] + 1.0)
        df['LWI_光水滋养'] = df['降水量'] * df['日照时数']

        # 路线A: 制表用（直观特征集）
        self.df_table = df.copy()
        round_cols = ['THI_温湿胁迫', 'PRI_抗药性预警', 'LWI_光水滋养']
        self.df_table[round_cols] = self.df_table[round_cols].round(2)

        # 路线B: ML标准化矩阵
        df_ml = df.copy()
        risk_map = {'低': 0, '中': 1, '高': 2}
        if '风险等级' in df_ml.columns:
            df_ml['风险等级_编码'] = df_ml['风险等级'].map(risk_map)
            if df_ml['风险等级_编码'].isna().any():
                df_ml['风险等级_编码'] = df_ml['风险等级_编码'].fillna(0).astype(int)

        # One-hot 编码分类变量
        categorical_cols = ['果树品种', '病虫害类型']
        for col in categorical_cols:
            if col in df_ml.columns:
                dummies = pd.get_dummies(df_ml[col], prefix=col)
                for c in dummies.columns:
                    if dummies[c].dtype == bool:
                        dummies[c] = dummies[c].astype(int)
                df_ml = pd.concat([df_ml, dummies], axis=1)
                df_ml.drop(columns=[col], inplace=True)

        # 移除所有剩余的非数值列（地块ID等）
        non_numeric_cols = df_ml.select_dtypes(exclude=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in non_numeric_cols if c != '风险等级编码']
        df_ml.drop(columns=non_numeric_cols, inplace=True, errors='ignore')

        # 数值特征标准化
        from sklearn.preprocessing import StandardScaler
        numeric_features = ['平均气温', '相对湿度', '降水量', '日照时数', '土壤湿度', '风速',
                            '近7天病株数', '近7天虫口密度', '近30天用药次数',
                            'THI_温湿胁迫', 'BTM_生物威胁动量', 'PRI_抗药性预警', 'LWI_光水滋养']
        numeric_features = [c for c in numeric_features if c in df_ml.columns]
        if numeric_features:
            scaler = StandardScaler()
            df_ml[numeric_features] = scaler.fit_transform(df_ml[numeric_features])

        # 清理
        drop_cols = ['地块ID', '风险等级']
        drop_cols = [c for c in drop_cols if c in df_ml.columns]
        df_ml.drop(columns=drop_cols, inplace=True, errors='ignore')

        # 确保风险等级_编码在最后一列
        if '风险等级_编码' in df_ml.columns:
            y_col = df_ml.pop('风险等级_编码')
            df_ml['风险等级_编码'] = y_col

        self.df_ml = df_ml
        self.n_samples = len(df_ml)
        self.n_classes = int(df_ml['风险等级_编码'].nunique()) if '风险等级_编码' in df_ml.columns else 3
        log(f"  制表数据: {self.df_table.shape}, ML矩阵: {self.df_ml.shape}")

    def _export(self):
        path_a = os.path.join(self.out_dir, '06_论文制表专用_直观特征集.csv')
        path_b = os.path.join(self.out_dir, '07_算法建模专用_数值标准矩阵.csv')
        self.df_table.to_csv(path_a, index=False, encoding='utf-8-sig')
        self.df_ml.to_csv(path_b, index=False, encoding='utf-8-sig')
        log(f"  已保存: {os.path.basename(path_a)}")
        log(f"  已保存: {os.path.basename(path_b)}")


# ===================================================================
#  阶段二: LightGBM 风险建模 + SHAP
# ===================================================================
class Stage2_RiskModeling:
    """基于 LightGBM 的多分类风险预警建模"""

    def __init__(self, df_ml: pd.DataFrame, df_table: pd.DataFrame):
        self.df_ml = df_ml
        self.df_table = df_table
        self.out_dir = os.path.join(OUTPUT_DIR, "2.低中高")
        self.fig_dir = os.path.join(self.out_dir, "figures")
        self.tab_dir = os.path.join(self.out_dir, "tables")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.tab_dir, exist_ok=True)

    def run(self):
        log("=" * 60)
        log("阶段二：LightGBM 风险建模")

        self._prepare_data()
        self._train_model()
        self._cross_validation()
        self._evaluate()
        self._feature_importance()
        self._shap_analysis()
        self._export()
        log("Stage 2 completed")
        return self

    def _prepare_data(self):
        from sklearn.model_selection import train_test_split
        target_col = "风险等级_编码"
        X = self.df_ml.drop(columns=[target_col])
        y = self.df_ml[target_col].astype(int)

        self.feature_names = X.columns.tolist()
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.n_classes = y.nunique()
        self.class_labels = sorted(y.unique().tolist())

        # 代价敏感权重
        self.class_weights = {
            k: self.n_samples / (self.n_classes * max((y == k).sum(), 1))
            for k in self.class_labels
        }
        sample_weights = y.map(self.class_weights).values

        self.X_train, self.X_test, self.y_train, self.y_test, self.sw_train, _ = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        log(f"  训练={self.X_train.shape[0]}, 测试={self.X_test.shape[0]}, 类别={self.n_classes}")

    def _train_model(self):
        import lightgbm as lgb
        self.lgb_params = {
            'objective': 'multiclass', 'num_class': self.n_classes,
            'metric': 'multi_logloss', 'boosting_type': 'gbdt',
            'num_leaves': 7, 'max_depth': 3,
            'learning_rate': 0.02, 'n_estimators': 300,
            'subsample': 0.6, 'colsample_bytree': 0.6,
            'reg_alpha': 2.0, 'reg_lambda': 3.0,
            'min_child_samples': 30,
            'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1,
        }
        self.model = lgb.LGBMClassifier(**self.lgb_params)
        self.model.fit(self.X_train, self.y_train, sample_weight=self.sw_train,
                       eval_set=[(self.X_test, self.y_test)],
                       callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        log("  模型训练完成")

    def _cross_validation(self):
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score, log_loss

        X = self.df_ml.drop(columns=["风险等级_编码"])
        y = self.df_ml["风险等级_编码"].astype(int)
        sample_weights = y.map(self.class_weights).values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_records = []

        for fold, (tr, val) in enumerate(skf.split(X, y), 1):
            fm = lgb.LGBMClassifier(**self.lgb_params)
            fm.fit(X.iloc[tr], y.iloc[tr], sample_weight=sample_weights[tr],
                   eval_set=[(X.iloc[val], y.iloc[val])],
                   callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            yp = fm.predict(X.iloc[val])
            ypb = fm.predict_proba(X.iloc[val])
            cv_records.append({
                'Fold': fold,
                'Accuracy': accuracy_score(y.iloc[val], yp),
                'Macro_F1': f1_score(y.iloc[val], yp, average='macro'),
                'Log_Loss': log_loss(y.iloc[val], ypb),
            })
            log(f"  Fold{fold} Acc={cv_records[-1]['Accuracy']:.4f} F1={cv_records[-1]['Macro_F1']:.4f}")

        self.cv_df = pd.DataFrame(cv_records)
        mean_acc = np.mean(self.cv_df['Accuracy'])
        mean_f1 = np.mean(self.cv_df['Macro_F1'])
        log(f"  CV均值: Acc={mean_acc:.4f}±{np.std(self.cv_df['Accuracy']):.4f}, "
            f"Macro-F1={mean_f1:.4f}±{np.std(self.cv_df['Macro_F1']):.4f}")

    def _evaluate(self):
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                      f1_score, confusion_matrix, classification_report,
                                      roc_curve, auc)
        from sklearn.preprocessing import label_binarize

        self.y_pred = self.model.predict(self.X_test)
        self.y_proba = self.model.predict_proba(self.X_test)

        self.per_class_precision = precision_score(self.y_test, self.y_pred, average=None)
        self.per_class_recall = recall_score(self.y_test, self.y_pred, average=None)
        self.per_class_f1 = f1_score(self.y_test, self.y_pred, average=None)
        self.macro_f1 = np.mean(self.per_class_f1)

        log(f"  高风险召回率={self.per_class_recall[-1]:.4f}")

        # 混淆矩阵
        self.cm = confusion_matrix(self.y_test, self.y_pred)

        # ROC AUC (OOF 五折预测)
        self._compute_oof_roc()

        # 分类报告
        self.cls_report = classification_report(self.y_test, self.y_pred,
                                                 target_names=[f'等级{k}' for k in self.class_labels],
                                                 output_dict=True)

    def _compute_oof_roc(self):
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        X = self.df_ml.drop(columns=["风险等级_编码"])
        y = self.df_ml["风险等级_编码"].astype(int)
        sample_weights = y.map(self.class_weights).values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof_proba = np.zeros((self.n_samples, self.n_classes))
        oof_true = np.zeros(self.n_samples)

        for tr, val in skf.split(X, y):
            fm = lgb.LGBMClassifier(**self.lgb_params)
            fm.fit(X.iloc[tr], y.iloc[tr], sample_weight=sample_weights[tr],
                   eval_set=[(X.iloc[val], y.iloc[val])],
                   callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            oof_proba[val] = fm.predict_proba(X.iloc[val])
            oof_true[val] = y.iloc[val].values

        self.oof_proba = oof_proba
        self.oof_true = oof_true.astype(int)
        oof_true_bin = label_binarize(self.oof_true, classes=self.class_labels)

        self.roc_auc_values = {}
        for i, k in enumerate(self.class_labels):
            if oof_true_bin[:, i].sum() == 0:
                self.roc_auc_values[f'等级{k}'] = 0.0
                continue
            fpr, tpr, _ = roc_curve(oof_true_bin[:, i], oof_proba[:, i])
            self.roc_auc_values[f'等级{k}'] = auc(fpr, tpr)
        self.macro_auc = np.mean(list(self.roc_auc_values.values()))
        log(f"  Macro-AUC(OOF)={self.macro_auc:.4f}")

    def _feature_importance(self):
        gain = self.model.booster_.feature_importance(importance_type='gain')
        split = self.model.booster_.feature_importance(importance_type='split')
        self.imp_df = pd.DataFrame({
            '特征': self.feature_names,
            'Gain重要性': gain,
            'Split次数': split,
        }).sort_values('Gain重要性', ascending=False).reset_index(drop=True)
        log(f"  Top3特征: {self.imp_df['特征'].head(3).tolist()}")

    def _shap_analysis(self):
        try:
            import shap
            explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent")
            shap_sample = self.X_test.iloc[:min(100, len(self.X_test))]
            shap_raw = explainer.shap_values(shap_sample)

            if isinstance(shap_raw, list):
                shap_list = [np.array(sv) for sv in shap_raw]
            elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
                shap_list = [shap_raw[:, :, i] for i in range(shap_raw.shape[2])]
            else:
                shap_list = [shap_raw]

            # 聚合 SHAP 贡献
            records = []
            for i in range(min(len(shap_list), self.n_classes)):
                mean_abs = np.abs(shap_list[i]).mean(axis=0)
                for j, feat in enumerate(self.feature_names):
                    records.append({
                        '风险等级': i,
                        '特征': feat,
                        'SHAP_mean_abs': mean_abs[j],
                    })
            self.shap_df = pd.DataFrame(records)
            log("  SHAP分析完成")
        except Exception as e:
            log(f"  SHAP跳过: {e}")
            self.shap_df = pd.DataFrame()

    def _export(self):
        # CV结果
        self.cv_df.to_csv(os.path.join(self.tab_dir, "CV_五折交叉验证.csv"), index=False, encoding='utf-8-sig')
        # 混淆矩阵
        cm_df = pd.DataFrame(self.cm,
                             index=[f'真实_等级{k}' for k in self.class_labels],
                             columns=[f'预测_等级{k}' for k in self.class_labels])
        cm_df.to_csv(os.path.join(self.tab_dir, "混淆矩阵.csv"), encoding='utf-8-sig')
        # ROC AUC
        auc_df = pd.DataFrame([
            {'风险等级': k, 'AUC_OOF': v} for k, v in self.roc_auc_values.items()
        ] + [{'风险等级': '宏平均', 'AUC_OOF': self.macro_auc}])
        auc_df.to_csv(os.path.join(self.tab_dir, "ROC_AUC值.csv"), index=False, encoding='utf-8-sig')
        # 特征重要性
        self.imp_df.to_csv(os.path.join(self.tab_dir, "特征重要性.csv"), index=False, encoding='utf-8-sig')
        # 类别分布
        y = self.df_ml["风险等级_编码"].astype(int)
        dist_df = pd.DataFrame({'风险等级': y.value_counts().index, '数量': y.value_counts().values})
        dist_df.to_csv(os.path.join(self.tab_dir, "类别分布.csv"), index=False, encoding='utf-8-sig')
        # SHAP
        if not self.shap_df.empty:
            self.shap_df.to_csv(os.path.join(self.tab_dir, "SHAP特征贡献.csv"), index=False, encoding='utf-8-sig')
        # KPI
        kpi_df = pd.DataFrame({
            '指标': ['Accuracy', 'Macro_F1', '高风险召回率', 'Macro_AUC_OOF'],
            '值': [
                np.mean(self.cv_df['Accuracy']),
                np.mean(self.cv_df['Macro_F1']),
                self.per_class_recall[-1] if len(self.per_class_recall) > 0 else 0,
                self.macro_auc,
            ]
        })
        kpi_df.to_csv(os.path.join(self.tab_dir, "核心KPI指标.csv"), index=False, encoding='utf-8-sig')
        # 测试集分类报告
        pd.DataFrame(self.cls_report).T.to_csv(os.path.join(self.tab_dir, "测试集分类报告.csv"), encoding='utf-8-sig')
        log("  所有表格已导出")


# ===================================================================
#  阶段三: 精准防控分析
# ===================================================================
class Stage3_PrecisionPrevention:
    """POSI + RRI + Jenks分区 + 帕累托优化 + 防控方案推荐"""

    def __init__(self, df_table: pd.DataFrame, df_ml: pd.DataFrame,
                 stage2: Stage2_RiskModeling):
        self.df_table = df_table
        self.df_ml = df_ml
        self.s2 = stage2
        self.out_dir = os.path.join(OUTPUT_DIR, "3.分区域分时段")
        self.fig_dir = os.path.join(self.out_dir, "figures")
        self.tab_dir = os.path.join(self.out_dir, "tables")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.tab_dir, exist_ok=True)

    def run(self):
        log("=" * 60)
        log("阶段三：精准防控分析")
        self._compute_full_probabilities()
        self._risk_zoning_rri_jenks()
        self._posi_window_detection()
        self._pareto_optimization()
        self._generate_recommendations()
        self._export_summary_tables()
        log("Stage 3 completed")
        return self

    def _compute_full_probabilities(self):
        """使用五折交叉预测计算全量地块风险概率"""
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold

        X = self.df_ml.drop(columns=["风险等级_编码"])
        y = self.df_ml["风险等级_编码"].astype(int)
        sample_weights = y.map(self.s2.class_weights).values
        n_samples = len(X)
        n_classes = self.s2.n_classes

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        all_proba = np.zeros((n_samples, n_classes))
        for tr, val in skf.split(X, y):
            fm = lgb.LGBMClassifier(**self.s2.lgb_params)
            fm.fit(X.iloc[tr], y.iloc[tr], sample_weight=sample_weights[tr],
                   eval_set=[(X.iloc[val], y.iloc[val])],
                   callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            all_proba[val] = fm.predict_proba(X.iloc[val])

        for k in range(n_classes):
            self.df_table[f'风险概率_等级{k}'] = all_proba[:, k]
        self.df_table['预测风险等级'] = np.argmax(all_proba, axis=1)
        self.df_table['预测风险标签'] = self.df_table['预测风险等级'].map({0: '低', 1: '中', 2: '高'})
        self.df_table['最大风险概率'] = all_proba.max(axis=1)
        risk_map = {'低': 0, '中': 1, '高': 2}
        if '风险等级' in self.df_table.columns:
            self.df_table['风险等级编码'] = self.df_table['风险等级'].map(risk_map)

        log(f"  预测分布: {dict(self.df_table['预测风险标签'].value_counts())}")

        # 保存全量地块风险概率
        self.df_table.to_csv(os.path.join(self.tab_dir, "00_全量地块风险概率与标签.csv"),
                             index=False, encoding='utf-8-sig')
        log("  已保存: 00_全量地块风险概率与标签.csv")

    def _risk_zoning_rri_jenks(self):
        """RRI + Jenks自然断点法分区"""
        n_classes = self.s2.n_classes
        class_weights = self.s2.class_weights
        w_raw = {k: class_weights[k] for k in range(n_classes)}
        w0 = w_raw[0]
        w_k = {k: round(w_raw[k] / w0, 4) for k in range(n_classes)}
        log(f"  归一化权重: {w_k}")

        group_cols = ['果树品种', '病虫害类型', '预测风险等级']
        zone_counts = self.df_table.groupby(group_cols).size().reset_index(name='地块数')

        rri_data = []
        for (variety, pest), grp in zone_counts.groupby(['果树品种', '病虫害类型']):
            rri = 0
            detail = {}
            for k in range(n_classes):
                sub = grp[grp['预测风险等级'] == k]
                n_k = sub['地块数'].values[0] if len(sub) > 0 else 0
                detail[f'等级{k}_地块数'] = n_k
                rri += w_k[k] * n_k
            rri_data.append({'果树品种': variety, '病虫害类型': pest, 'RRI': round(rri, 4), **detail})

        self.df_rri = pd.DataFrame(rri_data)
        rri_min, rri_max = self.df_rri['RRI'].min(), self.df_rri['RRI'].max()
        self.df_rri['RRI_star'] = ((self.df_rri['RRI'] - rri_min) / (rri_max - rri_min)
                                    if rri_max > rri_min else 0)
        self.df_rri = self.df_rri.sort_values('RRI_star', ascending=False).reset_index(drop=True)

        # Jenks 自然断点
        rri_values = self.df_rri['RRI_star'].values
        breaks_idx, self.GVF = self._jenks_natural_breaks(rri_values, k=3)
        rri_sorted = np.sort(rri_values)
        self.T1 = rri_sorted[breaks_idx[1]] if len(breaks_idx) > 2 else rri_sorted[-1]
        self.T2 = rri_sorted[breaks_idx[2]] if len(breaks_idx) > 2 else rri_sorted[-1]
        log(f"  Jenks断点: T1={self.T1:.4f}, T2={self.T2:.4f}, GVF={self.GVF:.4f}")

        def assign_zone(rri_star):
            if rri_star < self.T1:
                return '绿色区(常规监测)'
            elif rri_star < self.T2:
                return '黄色区(预防施药)'
            else:
                return '红色区(应急防控)'

        self.df_rri['防控响应区'] = self.df_rri['RRI_star'].apply(assign_zone)
        for zone in ['红色区(应急防控)', '黄色区(预防施药)', '绿色区(常规监测)']:
            n = len(self.df_rri[self.df_rri['防控响应区'] == zone])
            log(f"  {zone}: {n}个防控单元")

        self.df_rri.to_csv(os.path.join(self.tab_dir, "01_RRI区域风险指数与Jenks分区.csv"),
                           index=False, encoding='utf-8-sig')

    @staticmethod
    def _jenks_natural_breaks(values, k=3):
        """Jenks自然断点法"""
        values = np.array(sorted(values))
        n = len(values)
        if n <= k:
            return [0] + list(range(1, n)) + [n], 1.0

        cumsum = np.zeros(n + 1)
        cumsum2 = np.zeros(n + 1)
        for i in range(n):
            cumsum[i + 1] = cumsum[i] + values[i]
            cumsum2[i + 1] = cumsum2[i] + values[i] ** 2

        def ssd(start, end):
            s = cumsum[end] - cumsum[start]
            s2 = cumsum2[end] - cumsum2[start]
            m = end - start
            return s2 - (s ** 2) / m if m > 0 else 0

        dp = np.full((n + 1, k + 1), np.inf)
        dp[0, 0] = 0
        backtrack = np.zeros((n + 1, k + 1), dtype=int)

        for j in range(1, k + 1):
            for i in range(j, n + 1):
                for p in range(j - 1, i):
                    val = dp[p, j - 1] + ssd(p, i)
                    if val < dp[i, j]:
                        dp[i, j] = val
                        backtrack[i, j] = p

        breaks_idx = [n]
        i = n
        for j in range(k, 0, -1):
            i = backtrack[i, j]
            breaks_idx.append(i)
        breaks_idx = sorted(set(breaks_idx))

        total_mean = np.mean(values)
        SDAM = np.sum((values - total_mean) ** 2)
        SDCM = dp[n, k]
        GVF = 1 - (SDCM / SDAM) if SDAM > 0 else 1
        return breaks_idx, GVF

    def _posi_window_detection(self):
        """POSI病虫害发生适宜度指数 + 防治窗口判定"""
        # 特征重要性权重（从模型gain获取）
        gain_values = self.s2.model.booster_.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({'特征': self.s2.feature_names, 'Gain': gain_values})
        imp_df = imp_df.sort_values('Gain', ascending=False)

        # 7项POSI因子
        posi_factor_map = {
            '近7天病株数': {'tag': '病株', 'func': 'bio_threat'},
            '日照时数': {'tag': '日照', 'func': 'sunshine'},
            '降水量': {'tag': '降水', 'func': 'precip'},
            '近7天虫口密度': {'tag': '虫口', 'func': 'bio_threat'},
            '相对湿度': {'tag': '湿度', 'func': 'humidity'},
            'PRI_抗药性预警': {'tag': 'PRI', 'func': 'pri_inv'},
            'LWI_光水滋养': {'tag': 'LWI', 'func': 'lwi_synergy'},
        }

        posi_weights = {}
        total_gain = 0
        for feat, info in posi_factor_map.items():
            row = imp_df[imp_df['特征'] == feat]
            if len(row) > 0:
                g = row['Gain'].values[0]
                posi_weights[info['tag']] = max(g, 0.001)
                total_gain += posi_weights[info['tag']]

        for tag in posi_weights:
            posi_weights[tag] /= total_gain

        log(f"  POSI权重: { {k:round(v,3) for k,v in posi_weights.items()} }")

        # 隶属函数
        def f_bio_threat(B, B0=5, alpha=0.5):
            return 1.0 / (1.0 + np.exp(-alpha * (B - B0)))

        def f_sunshine(S, S_opt=6):
            return min(1.0, S / S_opt)

        def f_precipitation(P, P_opt=15, P_max=50):
            if P <= P_opt: return 1.0
            return max(0.0, 1.0 - (P - P_opt) / (P_max - P_opt))

        def f_humidity(H, H_opt=75, sigma=15):
            return np.exp(-((H - H_opt) ** 2) / (2 * sigma ** 2))

        # 计算POSI
        posi_values = []
        for _, row in self.df_table.iterrows():
            comp = {
                '病株': f_bio_threat(row['近7天病株数']),
                '日照': f_sunshine(row['日照时数']),
                '降水': f_precipitation(row['降水量']),
                '虫口': f_bio_threat(row['近7天虫口密度']),
                '湿度': f_humidity(row['相对湿度']),
                'PRI': 1.0 / (1.0 + row.get('PRI_抗药性预警', 1.0)),
                'LWI': min(f_precipitation(row['降水量']), f_sunshine(row['日照时数'])),
            }
            posi = sum(posi_weights.get(k, 0) * v for k, v in comp.items())
            posi_values.append(posi)

        self.df_table['POSI'] = posi_values
        self.theta_posi = np.percentile(posi_values, 70)
        self.df_table['在防治窗口内'] = self.df_table['POSI'] >= self.theta_posi

        n_in_window = self.df_table['在防治窗口内'].sum()
        log(f"  θ_posi={self.theta_posi:.4f}, 窗口内={n_in_window}/{len(self.df_table)}地块")

        self.posi_weights = posi_weights

        self.df_table.to_csv(os.path.join(self.tab_dir, "02_地块POSI与窗口判定.csv"),
                             index=False, encoding='utf-8-sig')

        # POSI因子权重表
        pw_df = pd.DataFrame([
            {'因子': tag, '权重': w} for tag, w in sorted(posi_weights.items(), key=lambda x: -x[1])
        ])
        pw_df.to_csv(os.path.join(self.tab_dir, "08_POSI因子权重.csv"), index=False, encoding='utf-8-sig')

        # 交叉统计表
        cross = pd.crosstab(self.df_table['预测风险标签'], self.df_table['在防治窗口内'])
        cross.to_csv(os.path.join(self.tab_dir, "表3-4_风险等级与防治窗口交叉统计.csv"), encoding='utf-8-sig')

    def _pareto_optimization(self):
        """多目标优化：ε-约束法求帕累托前沿"""
        n_samples = len(self.df_table)

        # 防控单元
        zone_units = []
        for (variety, pest, risk), grp in self.df_table.groupby(
                ['果树品种', '病虫害类型', '预测风险等级']):
            n_plots = len(grp)
            in_window = grp['在防治窗口内'].sum()
            zone_units.append({
                '果树品种': variety, '病虫害类型': pest, '风险等级': risk,
                '地块数': n_plots, '窗口内地块数': in_window,
                '平均POSI': round(grp['POSI'].mean(), 4),
            })
        self.df_units = pd.DataFrame(zone_units)

        # 简化帕累托计算
        T_season = 180
        dt_baseline = 10
        n_app_baseline = T_season / dt_baseline
        Q_baseline = n_samples * 1.0 * n_app_baseline

        pareto_front = []
        for mid_dr in [0.3, 0.5, 0.7, 0.9, 1.0]:
            for high_dr in [0.5, 0.7, 1.0]:
                total_dose = 0
                for _, row in self.df_units.iterrows():
                    k = int(row['风险等级'])
                    n_win = row['窗口内地块数']
                    dr = {0: 0, 1: mid_dr, 2: high_dr}[k]
                    n_app = n_app_baseline
                    total_dose += n_win * dr * n_app

                eff = total_dose / Q_baseline if Q_baseline > 0 else 0
                pareto_front.append({
                    'Q_used': total_dose, 'Efficacy': 1 - eff,
                    'mid_dose': mid_dr, 'high_dose': high_dr,
                    'window_only': True,
                })

        self.df_pareto = pd.DataFrame(pareto_front).drop_duplicates(
            subset=['Q_used', 'Efficacy']).sort_values('Q_used')
        self.Q_baseline = Q_baseline
        self.PRR = (Q_baseline - self.df_pareto['Q_used'].min()) / Q_baseline * 100 if len(
            self.df_pareto) > 0 else 0
        log(f"  帕累托前沿: {len(self.df_pareto)}个解, 最高减药率≈{self.PRR:.1f}%")

        if len(self.df_pareto) > 0:
            self.df_pareto.to_csv(os.path.join(self.tab_dir, "04_帕累托前沿_减药增效权衡.csv"),
                                  index=False, encoding='utf-8-sig')

    def _generate_recommendations(self):
        """为每个防控单元生成精准防控推荐方案"""
        recs = []
        for _, row in self.df_units.iterrows():
            k = int(row['风险等级'])
            n = row['地块数']
            n_win = row['窗口内地块数']

            if k == 0:
                strategy, dose, pesticides, freq = '监测为主', 0, '无（待命）', '每7天巡查1次'
            elif k == 1:
                strategy, dose, pesticides, freq = '预防施药', '待优化', '杀虫剂B → 生物农药C 轮换', f'窗口期每7天1次'
            else:
                strategy, dose, pesticides, freq = '应急防控', 100, '杀菌剂A → 杀虫剂B → 生物农药C 三轮换', '立即响应全覆盖'

            recs.append({
                '果树品种': row['果树品种'], '病虫害类型': row['病虫害类型'],
                '风险等级': k, '地块数': n, '窗口内地块数': n_win,
                '防控策略': strategy, '推荐剂量(%)': dose,
                '推荐农药轮换': pesticides, '施药频率': freq,
            })

        self.df_recs = pd.DataFrame(recs)
        self.df_recs.to_csv(os.path.join(self.tab_dir, "05_防控单元精准方案推荐.csv"),
                            index=False, encoding='utf-8-sig')
        log(f"  已生成{len(self.df_recs)}条防控方案推荐")

    def _export_summary_tables(self):
        """导出汇总表"""
        # 三级防控响应区汇总
        zone_summary = self.df_rri.groupby('防控响应区').agg(
            防控单元数=('防控响应区', 'count'),
            平均RRI=('RRI_star', 'mean'),
        ).reset_index()
        zone_summary['平均RRI'] = zone_summary['平均RRI'].round(4)
        zone_summary.to_csv(os.path.join(self.tab_dir, "06_防控响应区汇总统计.csv"),
                            index=False, encoding='utf-8-sig')

        # 三级防控响应区汇总（表3-1格式）
        zone_summary.to_csv(os.path.join(self.tab_dir, "表3-1_三级防控响应区汇总.csv"),
                            index=False, encoding='utf-8-sig')

        # 三级差异化防控策略体系（表3-5）
        strategy_df = pd.DataFrame([
            {'响应区': '绿色区(常规监测)', '策略': '常规监测', '施药': '无需施药', '巡查频率': '每7天1次'},
            {'响应区': '黄色区(预防施药)', '策略': '预防施药', '施药': '窗口期内点状施药(减量)', '巡查频率': '每3天1次'},
            {'响应区': '红色区(应急防控)', '策略': '应急防控', '施药': '立即全覆盖施药(全量)', '巡查频率': '每天1次'},
        ])
        strategy_df.to_csv(os.path.join(self.tab_dir, "表3-5_三级差异化防控策略体系.csv"),
                           index=False, encoding='utf-8-sig')

        # 防控单元精准方案推荐精简版（表3-6）
        if len(self.df_recs) > 0:
            compact = self.df_recs[['果树品种', '病虫害类型', '风险等级', '防控策略', '施药频率']].head(50)
            compact.to_csv(os.path.join(self.tab_dir, "表3-6_防控单元精准方案推荐_精简版.csv"),
                           index=False, encoding='utf-8-sig')

        # 地块级详情前50
        detail_cols = ['地块ID', '果树品种', '病虫害类型', '预测风险标签',
                       'POSI', '在防治窗口内', '最大风险概率']
        detail_cols = [c for c in detail_cols if c in self.df_table.columns]
        self.df_table[detail_cols].head(50).to_csv(
            os.path.join(self.tab_dir, "07_地块级详情_前50.csv"), index=False, encoding='utf-8-sig')

        log("  汇总表全部导出完成")


# ===================================================================
#  主管线入口
# ===================================================================
def run_pipeline(input_csv: str) -> dict:
    """
    执行完整三阶段数据处理管线

    参数:
        input_csv: 原始数据集CSV路径

    返回:
        dict: {'success': bool, 'message': str, 'stats': dict}
    """
    start_time = time.time()
    stats = {}

    try:
        # 阶段一
        s1 = Stage1_Preprocessing(input_csv).run()
        stats['n_samples'] = s1.n_samples
        stats['n_classes'] = s1.n_classes
        stats['features'] = list(s1.df_ml.columns[:5]) + ['...']

        # 阶段二
        s2 = Stage2_RiskModeling(s1.df_ml, s1.df_table).run()
        stats['cv_accuracy'] = round(np.mean(s2.cv_df['Accuracy']), 4)
        stats['cv_macro_f1'] = round(np.mean(s2.cv_df['Macro_F1']), 4)
        stats['macro_auc'] = round(s2.macro_auc, 4)

        # 阶段三
        s3 = Stage3_PrecisionPrevention(s1.df_table, s1.df_ml, s2).run()
        stats['n_zone_units'] = len(s3.df_rri)
        stats['PRR'] = round(s3.PRR, 1)
        stats['theta_posi'] = round(s3.theta_posi, 4)

        elapsed = time.time() - start_time
        stats['elapsed'] = f"{elapsed:.1f}s"

        log("=" * 60)
        log("Pipeline completed! Total time: " + f"{elapsed:.1f}s")
        log(f"   Samples: {stats['n_samples']}, CV-F1: {stats['cv_macro_f1']}, AUC: {stats['macro_auc']}")

        return {'success': True, 'message': '管线执行成功', 'stats': stats}

    except Exception as e:
        elapsed = time.time() - start_time
        log(f"Pipeline failed: {e}")
        traceback.print_exc()
        return {'success': False, 'message': str(e), 'stats': {'elapsed': f"{elapsed:.1f}s"}}


# ==================== CLI入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="果园病虫害风险预警数据处理管线")
    parser.add_argument("--input", type=str, default="data_process/data.csv",
                        help="原始数据集CSV路径")
    args = parser.parse_args()

    result = run_pipeline(args.input)
    print(f"\n结果: {result['message']}")
    if result['success']:
        print(f"统计: {result['stats']}")
