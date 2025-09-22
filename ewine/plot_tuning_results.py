import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_tuning_heatmaps():
    """
    读取完整的超参数调优结果CSV文件，并为每个模型生成热图。
    """
    # --- 路径和文件名配置 ---
    project_dir = Path(__file__).resolve().parent
    results_csv = project_dir / "Tuning_Runs" / "hyperparameter_tuning_full_results.csv"
    plot_dir = project_dir / "Tuning_Runs" / "plots"
    plot_dir.mkdir(exist_ok=True)

    # --- 检查文件是否存在 ---
    if not results_csv.exists():
        print(f"❌ 错误: 结果文件未找到！")
        print(f"   请先运行 'run_hyperparameter_tuning.py' 来生成: {results_csv}")
        return

    print(f"📊 正在从 {results_csv} 加载数据并生成热图...")
    full_df = pd.read_csv(results_csv)

    # --- 为每个模型生成一个热图 ---
    for model_name in full_df['Model'].unique():
        model_df = full_df[full_df['Model'] == model_name].copy()
        
        # 找出该模型的超参数列
        param_cols = [col for col in model_df.columns if col not in ['Model', 'Score']]
        
        if len(param_cols) != 2:
            print(f"  - 警告: 模型 '{model_name}' 的超参数数量不为2 ({len(param_cols)}个)，无法绘制2D热图。已跳过。")
            continue

        # 使用 pivot_table 将数据重塑为网格状，以便绘制热图
        param1, param2 = param_cols[0], param_cols[1]
        try:
            heatmap_data = model_df.pivot_table(
                index=param1, 
                columns=param2, 
                values='Score'
            )

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                heatmap_data, 
                annot=True,      # 在每个单元格上显示数值
                fmt=".4f",       # 数值格式化为4位小数
                cmap="viridis",  # 使用 'viridis' 色彩映射，颜色越亮性能越好
                linewidths=.5
            )
            
            # 清理参数名以用作标题
            param1_clean = param1.replace("--","").replace("_", " ").title()
            param2_clean = param2.replace("--","").replace("_", " ").title()
            
            plt.title(f'Hyperparameter Tuning for {model_name}\n(Score: Balanced Accuracy CV Mean)')
            plt.xlabel(param2_clean)
            plt.ylabel(param1_clean)
            plt.tight_layout()

            # 保存图像
            plot_path = plot_dir / f"heatmap_{model_name}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"  ✅ 成功为 '{model_name}' 生成热图，已保存到: {plot_path.name}")

        except Exception as e:
            print(f"  ❌ 错误: 为 '{model_name}' 生成热图时失败: {e}")

    print("\n✨ 可视化完成！")

if __name__ == "__main__":
    plot_tuning_heatmaps()
