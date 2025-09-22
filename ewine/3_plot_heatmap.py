# -*- coding: utf-8 -*-
"""
独立的热力图生成脚本 (Standalone Heatmap Plotter)

功能:
- 从 ablation_summary_table.csv 读取数据。
- 筛选出与正则化强度网格搜索相关的数据。
- 绘制 RMSE (或其他指标) vs. 正则化强度的热力图。
- 允许通过命令行参数轻松自定义图形的各个方面 (字体、颜色、标题等)。

使用方法:
python plot_heatmap.py --input <path_to_csv> --output <image_path.png> [自定义选项]
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_custom_heatmap(
    input_csv: Path,
    output_png: Path,
    metric_col: str,
    title: str,
    xlabel: str,
    ylabel:str,
    cmap: str,
    font_size: int,
    annot_fmt: str
):
    """
    读取分析结果的CSV文件并生成可定制的热力图。
    """
    if not input_csv.exists():
        print(f"❌ 错误: 输入文件不存在 -> {input_csv}")
        return

    print(f"🔄 正在读取数据从: {input_csv}")
    # 读取时将第一列作为索引
    df = pd.read_csv(input_csv, index_col=0)

    # 1. 筛选出正则化网格搜索的结果
    reg_df = df[df.index.str.startswith("RegGrid")].copy()
    if reg_df.empty:
        print("❌ 错误: 在CSV中未找到 'RegGrid' 相关的数据行。")
        return

    # 2. 从索引中提取 lambda_mask 和 lambda_ga 的值
    try:
        idx_series = reg_df.index.to_series()
        reg_df['lambda_mask'] = idx_series.str.extract(r'mask([\d.]+)')[0].astype(float)
        reg_df['lambda_ga'] = idx_series.str.extract(r'ga([\d.]+)')[0].astype(float)
    except Exception as e:
        print(f"❌ 错误: 从索引解析 lambda 值失败: {e}")
        print("   请确保索引格式为 'RegGrid_maskX.X_gaY.Y'")
        return

    # 3. 检查指定的指标列是否存在
    if metric_col not in reg_df.columns:
        print(f"❌ 错误: 指定的指标列 '{metric_col}' 不存在。")
        print(f"   可用列包括: {list(reg_df.columns)}")
        return

    # 4. 创建数据透视表 (Pivot Table)
    print(f"📊 正在为指标 '{metric_col}' 创建热力图...")
    heatmap_df = reg_df.pivot(index='lambda_mask', columns='lambda_ga', values=metric_col)
    
    # 5. 开始绘图
    # 设置全局字体大小
    mpl.rcParams.update({'font.size': font_size})
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    
    # 使用 imshow 绘制热力图
    im = ax.imshow(heatmap_df.values, cmap=cmap)

    # 添加颜色条 (Colorbar)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric_col, rotation=-90, va="bottom")

    # 设置坐标轴刻度和标签
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_xticklabels(heatmap_df.columns)
    ax.set_yticklabels(heatmap_df.index)
    
    # 设置坐标轴标题和图表总标题
    ax.set_xlabel(xlabel, fontsize=font_size + 2)
    ax.set_ylabel(ylabel, fontsize=font_size + 2)
    ax.set_title(title, fontsize=font_size + 4, pad=20)

    # 在每个单元格中添加数值标注
    # 为了保证可读性，根据背景颜色深浅自动切换文字颜色
    threshold = im.norm(heatmap_df.values.max()) / 2.
    textcolors = ("black", "white")
    
    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            val = heatmap_df.values[i, j]
            # 如果值是 NaN 则不显示
            if pd.isna(val): continue
            color = textcolors[int(im.norm(val) < threshold)]
            ax.text(j, i, format(val, annot_fmt),
                    ha="center", va="center", color=color, fontsize=font_size - 1)

    # 优化布局并保存
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    print(f"✅ 热力图已成功保存到 -> {output_png}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="从消融研究的汇总CSV文件生成热力图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 核心参数
    parser.add_argument(
        "--input", "-i", type=Path,
        default="Ablation_Study_Runs_Advanced/ablation_summary_table.csv",
        help="输入的CSV文件路径"
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default="Ablation_Study_Runs_Advanced/heatmap_RMSE_vs_regularizers_custom.png",
        help="输出的PNG图片文件路径"
    )
    parser.add_argument(
        "--metric", "-m", type=str,
        default="rmse_mm_cv_mean_mean",
        help="要在热力图上展示的指标列名"
    )
    # 图形定制参数
    parser.add_argument("--title", type=str, default="Heatmap of RMSE vs. Regularizer Strengths", help="图表标题")
    parser.add_argument("--xlabel", type=str, default="Gradient-Alignment Strength (lambda_ga)", help="X轴标签")
    parser.add_argument("--ylabel", type=str, default="Physics-Mask Strength (lambda_mask)", help="Y轴标签")
    parser.add_argument("--cmap", type=str, default="viridis_r", help="Matplotlib 调色板 (e.g., 'viridis_r', 'coolwarm', 'YlGnBu')")
    parser.add_argument("--font_size", type=int, default=12, help="基础字体大小")
    parser.add_argument("--annot_fmt", type=str, default=".4f", help="单元格内数值的格式化字符串 (e.g., '.2f', '.3e')")

    args = parser.parse_args()

    plot_custom_heatmap(
        input_csv=args.input,
        output_png=args.output,
        metric_col=args.metric,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        cmap=args.cmap,
        font_size=args.font_size,
        annot_fmt=args.annot_fmt
    )


if __name__ == "__main__":
    main()