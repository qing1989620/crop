import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# 全局学术图表规范设置 (Publication-ready)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="ticks", font='SimHei')
sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 2.5})

class OrchardDataProcessor:
    def __init__(self, input_path, output_base_dir):
        self.input_path = input_path
        self.output_dir = os.path.join(output_base_dir, "预处理及特征工程结果")
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None
        self.df_table_ready = None
        self.df_ml_ready = None
        print(f"🔧 初始化完成，多维学术图表输出目录: {self.output_dir}")

    def load_data(self):
        print("📥 正在加载原始数据集...")
        self.df = pd.read_csv(self.input_path, encoding='gbk')

    def plot_pre_processing_eda(self):
        """基于全排头维度的探索性分析 (分类交叉图 + 多维雷达图)"""
        print("📊 正在生成 [预处理前] 全维度排头交叉与雷达图...")
        
        # --- 1. 基于分类排头的交叉分析 ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.countplot(data=self.df, x='果树品种', hue='风险等级', 
                      hue_order=['低', '中', '高'], palette='YlOrRd', edgecolor='black', ax=axes[0])
        axes[0].set_title('各果树品种面临的风险等级结构', fontsize=16, fontweight='bold', pad=15)
        axes[0].set_ylabel('地块数量 (Count)')
        axes[0].legend(title='风险等级')

        sns.countplot(data=self.df, x='病虫害类型', hue='风险等级', 
                      hue_order=['低', '中', '高'], palette='magma_r', edgecolor='black', ax=axes[1])
        axes[1].set_title('各病虫害类型引发的风险等级严重度', fontsize=16, fontweight='bold', pad=15)
        axes[1].set_ylabel('地块数量 (Count)')
        axes[1].legend(title='风险等级')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_原数据_排头分类特征交叉图.png'), dpi=400)
        plt.close()

        # --- 2. 基于连续环境排头的生态雷达图 ---
        env_cols = ['平均气温', '相对湿度', '降水量', '日照时数', '土壤湿度', '风速']
        radar_data = self.df.groupby('风险等级')[env_cols].mean().reindex(['低', '中', '高'])
        
        scaler = MinMaxScaler(feature_range=(0.1, 1))
        radar_scaled = pd.DataFrame(scaler.fit_transform(radar_data), 
                                    columns=env_cols, index=radar_data.index)

        angles = np.linspace(0, 2 * np.pi, len(env_cols), endpoint=False).tolist()
        angles += angles[:1] 
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = {'低': '#2ca02c', '中': '#ff7f0e', '高': '#d62728'}
        
        for risk in ['低', '中', '高']:
            values = radar_scaled.loc[risk].tolist()
            values += values[:1]
            ax.plot(angles, values, color=colors[risk], linewidth=2, label=f'{risk}风险地块均值')
            ax.fill(angles, values, color=colors[risk], alpha=0.15)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), env_cols, fontsize=13, fontweight='bold')
        ax.set_title('不同风险等级下多维环境生态指纹雷达图', fontsize=18, fontweight='bold', pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.savefig(os.path.join(self.output_dir, '02_原数据_六维环境特征雷达图.png'), dpi=400, bbox_inches='tight')
        plt.close()

    def feature_engineering_and_preprocess(self):
        """数学建模化特征工程与双轨制数据导出"""
        print("⚙️ 正在执行高阶特征派生与双轨制数据分离...")
        df_base = self.df.copy()

        df_base['THI_温湿胁迫'] = df_base['平均气温'] * df_base['相对湿度']
        df_base['BTM_生物威胁动量'] = df_base['近7天病株数'] + df_base['近7天虫口密度']
        df_base['PRI_抗药性预警'] = df_base['BTM_生物威胁动量'] / (df_base['近30天用药次数'] + 1.0)
        df_base['LWI_光水滋养'] = df_base['降水量'] * df_base['日照时数']

        # 路线A：制表用数据
        self.df_table_ready = df_base.copy()
        round_cols = ['THI_温湿胁迫', 'PRI_抗药性预警', 'LWI_光水滋养']
        self.df_table_ready[round_cols] = self.df_table_ready[round_cols].round(2)
        cols_order = ['地块ID', '果树品种', '病虫害类型', '风险等级', 'THI_温湿胁迫', 'BTM_生物威胁动量', 'PRI_抗药性预警']
        remaining_cols = [c for c in self.df_table_ready.columns if c not in cols_order]
        self.df_table_ready = self.df_table_ready[cols_order + remaining_cols]

        # 路线B：算法建模用数据
        df_ml = df_base.copy()
        df_ml['风险等级_编码'] = df_ml['风险等级'].map({'低': 0, '中': 1, '高': 2})
        df_ml = pd.get_dummies(df_ml, columns=['果树品种', '病虫害类型'], prefix=['品种', '病虫害'])
        for col in df_ml.columns:
            if df_ml[col].dtype == bool:
                df_ml[col] = df_ml[col].astype(int)

        numeric_features = ['平均气温', '相对湿度', '降水量', '日照时数', '土壤湿度', '风速', 
                            '近7天病株数', '近7天虫口密度', '近30天用药次数',
                            'THI_温湿胁迫', 'BTM_生物威胁动量', 'PRI_抗药性预警', 'LWI_光水滋养']
        scaler = StandardScaler()
        df_ml[numeric_features] = scaler.fit_transform(df_ml[numeric_features])
        
        df_ml.drop(columns=['地块ID', '风险等级'], inplace=True)
        y_col = df_ml.pop('风险等级_编码')
        df_ml['风险等级_编码'] = y_col
        self.df_ml_ready = df_ml
        
    def plot_post_processing_eda(self):
        """高阶特征的深层折线与密度验证"""
        print("📈 正在生成 [衍生特征] 演化折线图与相关性图表...")
        
        # --- 1. 高阶特征单调递增折线图 (带95%置信区间) ---
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = self.df_ml_ready[['THI_温湿胁迫', 'BTM_生物威胁动量', 'PRI_抗药性预警', '风险等级_编码']].copy()
        plot_df['风险等级'] = plot_df['风险等级_编码'].map({0: '低', 1: '中', 2: '高'})
        
        sns.pointplot(data=plot_df, x='风险等级', y='THI_温湿胁迫', order=['低', '中', '高'], 
                      color='#1f77b4', markers='o', label='THI_温湿胁迫', err_kws={'linewidth': 1.5})
        sns.pointplot(data=plot_df, x='风险等级', y='BTM_生物威胁动量', order=['低', '中', '高'], 
                      color='#ff7f0e', markers='s', label='BTM_生物威胁动量', err_kws={'linewidth': 1.5})
        sns.pointplot(data=plot_df, x='风险等级', y='PRI_抗药性预警', order=['低', '中', '高'], 
                      color='#d62728', markers='^', label='PRI_抗药性预警', err_kws={'linewidth': 1.5})
        
        ax.set_title('核心衍生指标随风险等级演化的趋势折线图 (Z-score 标准化)', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('标准化特征值 (Mean with 95% CI)', fontsize=14)
        ax.set_xlabel('地块最终风险等级', fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="高阶衍生特征", loc='upper left')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_处理后_衍生特征演化趋势折线图.png'), dpi=400)
        plt.close()

        # --- 2. 高阶特征核密度与蜂群散点图 ---
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle('核心衍生农学特征在不同风险等级下的分布穿透分析', fontsize=20, fontweight='bold', y=1.05)

        def plot_violin_swarm(ax_obj, y_col, title, palette):
            sns.violinplot(x='风险等级', y=y_col, data=self.df_table_ready, 
                           order=['低', '中', '高'], hue='风险等级', palette=palette, 
                           inner="quartile", alpha=0.5, ax=ax_obj, legend=False)
            sns.swarmplot(x='风险等级', y=y_col, data=self.df_table_ready, 
                          order=['低', '中', '高'], hue='风险等级', palette=palette, 
                          edgecolor="black", linewidth=0.8, size=4.5, ax=ax_obj, legend=False)
            ax_obj.set_title(title, fontsize=16, pad=15)
            ax_obj.set_ylabel(f'真实物理量: {y_col.split("_")[0]}', fontsize=14)
            ax_obj.set_xlabel('')

        plot_violin_swarm(axes[0], 'PRI_抗药性预警', 'PRI (药效抗药性预警指数) 核密度分布', 'Reds')
        plot_violin_swarm(axes[1], 'THI_温湿胁迫', 'THI (温湿协同胁迫) 核密度分布', 'Blues')
        plot_violin_swarm(axes[2], 'BTM_生物威胁动量', 'BTM (生物威胁复合动量) 核密度分布', 'Oranges')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_处理后_高阶特征机理小提琴图.png'), dpi=400, bbox_inches='tight')
        plt.close()

        # --- 3. 全局多维相关性热力图 ---
        plt.figure(figsize=(11, 9))
        corr_cols = ['平均气温', '相对湿度', '降水量', 'THI_温湿胁迫', 'BTM_生物威胁动量', 
                     'PRI_抗药性预警', 'LWI_光水滋养', '近30天用药次数', '风险等级_编码']
        corr_matrix = self.df_ml_ready[corr_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', fmt=".2f", 
                    linewidths=1.5, linecolor='white', center=0, 
                    square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 11, "weight": "bold"})
        plt.title('全局数值型空间多维相关性测度矩阵', fontsize=18, fontweight='bold', pad=20)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_处理后_全局特征相关性热力图.png'), dpi=400)
        plt.close()

    def export_data(self):
        print("💾 正在输出高质量制表文件与机器学习矩阵...")
        table_path = os.path.join(self.output_dir, '06_论文制表专用_直观特征集.csv')
        self.df_table_ready.to_csv(table_path, index=False, encoding='gbk')
        
        ml_path = os.path.join(self.output_dir, '07_算法建模专用_数值标准矩阵.csv')
        self.df_ml_ready.to_csv(ml_path, index=False, encoding='gbk')
        
        print(f"✅ [直观版数据] 用于插入论文制表: {table_path}")
        print(f"✅ [编码版数据] 用于训练 XGBoost/LightGBM: {ml_path}")

    def run_pipeline(self):
        self.load_data()
        self.plot_pre_processing_eda()
        self.feature_engineering_and_preprocess()
        self.plot_post_processing_eda()
        self.export_data()
        print("🎉 恭喜！包含雷达图与折线图的终极分析报告生成完毕。")

if __name__ == "__main__":
    INPUT_FILE = r"D:\qing_zhuomian_\工作区\数据要素\data.csv"
    OUTPUT_BASE = r"D:\qing_zhuomian_\工作区\数据要素\output"
    
    processor = OrchardDataProcessor(input_path=INPUT_FILE, output_base_dir=OUTPUT_BASE)
    processor.run_pipeline()