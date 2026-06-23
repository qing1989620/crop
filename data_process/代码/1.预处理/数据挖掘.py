import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# ==========================================
# 全局学术图表规范设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="ticks", font='SimHei')
sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 2})

class BaselineModelEvaluator:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        # 复用之前的输出文件夹，保持工程结构整洁
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None
        
    def load_and_split_data(self):
        print("📥 正在加载标准化算法矩阵...")
        self.df = pd.read_csv(self.data_path, encoding='gbk')
        
        # 分离特征矩阵 X 和 标签向量 y
        X = self.df.drop(columns=['风险等级_编码'])
        y = self.df['风险等级_编码']
        
        # 严谨的学术做法：必须在训练集上评估特征重要性，防止数据泄露
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✅ 数据切分完成 | 训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}")

    def train_baseline_and_extract_importance(self):
        print("⚙️ 正在训练 LightGBM 基线模型并计算 Information Gain...")
        
        # 初始化 LightGBM 分类器
        # importance_type='gain' 意味着使用信息增益来评估重要性，比 'split' (分裂次数) 更具科学性
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            random_state=42,
            importance_type='gain',
            n_estimators=100,
            learning_rate=0.05,
            verbose=-1  # 静默模式，不输出 Warning，不影响模型结果
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # 提取特征重要性并构建 DataFrame
        importance_values = self.model.feature_importances_
        feature_names = self.X_train.columns
        
        self.importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance_Gain': importance_values
        })
        
        # 按重要性降序排列，并计算相对重要性百分比
        self.importance_df = self.importance_df.sort_values(by='Importance_Gain', ascending=False).reset_index(drop=True)
        self.importance_df['Relative_Importance_(%)'] = 100.0 * (self.importance_df['Importance_Gain'] / self.importance_df['Importance_Gain'].sum())
        
        print("✅ 特征重要性量化评估完成！")

    def plot_feature_importance(self):
        print("📊 正在生成学术级特征重要性排序图...")
        
        plt.figure(figsize=(12, 8))
        
        # 取排名前 15 的特征进行展示（防止特征过多图表拥挤）
        top_n = min(15, len(self.importance_df))
        plot_data = self.importance_df.head(top_n)
        
        # 使用 seaborn 绘制水平条形图，采用渐变色映射突出头部特征
        ax = sns.barplot(
            x='Relative_Importance_(%)', 
            y='Feature', 
            data=plot_data,
            hue='Feature',
            palette='viridis',  # 科学界推崇的感知均匀色板
            legend=False
        )
        
        # 在条形图尾部添加精确数值标签
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.5, p.get_y() + p.get_height() / 2.,
                    f'{width:.1f}%', ha='left', va='center', 
                    fontsize=12, fontweight='bold', color='#333333')
            
        plt.title('基于 LightGBM (Information Gain) 的全局特征重要性评估', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('相对重要性贡献率 / Relative Importance (%)', fontsize=14, labelpad=10)
        plt.ylabel('输入特征 / Features', fontsize=14, labelpad=10)
        
        # 扩展 x 轴上限，防止数值标签被边缘裁切
        plt.xlim(0, plot_data['Relative_Importance_(%)'].max() * 1.15)
        
        # 增加网格线以便对齐查看
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, '08_基线模型_全局特征重要性排序图.png')
        plt.savefig(save_path, dpi=400)
        plt.close()
        print(f"✅ 图表已保存至: {save_path}")

    def export_importance_report(self):
        # 导出为 CSV，方便你在论文中制作特征贡献表
        save_path = os.path.join(self.output_dir, '09_基线模型_特征重要性量化报表.csv')
        self.importance_df.to_csv(save_path, index=False, encoding='gbk')
        print(f"✅ 报表已保存至: {save_path}")

    def run(self):
        self.load_and_split_data()
        self.train_baseline_and_extract_importance()
        self.plot_feature_importance()
        self.export_importance_report()
        print("🎉 赛题任务 1 (挖掘关键影响特征) 彻底完美收官！")

if __name__ == "__main__":
    # 指向我们刚刚生成的标准矩阵
    DATA_FILE = r"D:\qing_zhuomian_\工作区\数据要素\output\预处理及特征工程结果\07_算法建模专用_数值标准矩阵.csv"
    # 复用同一个输出目录
    OUTPUT_DIR = r"D:\qing_zhuomian_\工作区\数据要素\output\预处理及特征工程结果"
    
    evaluator = BaselineModelEvaluator(data_path=DATA_FILE, output_dir=OUTPUT_DIR)
    evaluator.run()