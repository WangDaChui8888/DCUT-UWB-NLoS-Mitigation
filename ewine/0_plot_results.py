# -*- coding: utf-8 -*-
"""
0_plot_results.py — Publication-ready figures (complete version)

Features
- Read CSV summary with means, 95% CIs, Holm/FDR-adjusted p-values
- Grouped bar charts for classification (4 metrics) and regression (RMSE/MAE)
- Significance stars (based on Holm-adjusted p-values) on regression bars vs main model
- ROC/PR curves with per-seed mean and t-based 95% CI shading
- Error CDF curves with per-seed mean and t-based 95% CI shading
- One-click PNG export (600 DPI) into ./Ghent_Statistical_Runs/plots

Directory assumptions
- Results root: ./Ghent_Statistical_Runs
- Summary CSV:  Ghent_Statistical_Runs/final_statistical_summary_with_pvalues.csv
- Per-seed .mat: Ghent_Statistical_Runs/run_seed_XXXX/<files named by config*.py>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.ticker import MultipleLocator, FuncFormatter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from scipy import stats
from scipy.io import loadmat

# ======================= Paths & global settings =======================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_RESULTS_DIR = SCRIPT_DIR / "Ghent_Statistical_Runs"
SUMMARY_CSV = BASE_RESULTS_DIR / "final_statistical_summary_with_pvalues.csv"
PLOTS_DIR = BASE_RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 主模型名（用于从 CSV 读取 p 值列）。若你的 CSV 里主模型名为 "DCUT"，改为 "DCUT"
MAIN_MODEL_KEY_FOR_CSV = "DualTransformer"

# 这些键用于从 .mat 加载不同模型的结果（不影响条形图）
MODEL_KEYS_FOR_MAT = ["DualTransformer", "DNN", "SingleTransformer", "SVM", "XGBoost", "CNNLSTM"]

# 展示名（图例）
DISPLAY_MAP: Dict[str, str] = {
    "DualTransformer": "DCUT",
    "DNN": "DNN",
    "SingleTransformer": "ST",
    "SVM": "LS-SVM",
    "XGBoost": "XGBoost",
    "CNNLSTM": "CNN-LSTM",
}

# 调色板与线型（曲线图使用；条形图颜色在函数内定义）
PALETTE: Dict[str, str] = {
    "DualTransformer": "#2E86AB",
    "DNN": "#A23B72",
    "SingleTransformer": "#F18F01",
    "SVM": "#C73E1D",
    "XGBoost": "#27AE60",
    "CNNLSTM": "#8E44AD",
    "Default": "#666666",
}
LINESTYLE: Dict[str, object] = {
    "DualTransformer": "-",
    "DNN": "--",
    "SingleTransformer": "-.",
    "SVM": ":",
    "XGBoost": (0, (5, 3)),   # loosely dashed
    "CNNLSTM": (0, (1, 1)),   # densely dotted
    "Default": "-",
}

# Matplotlib 全局样式
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize":14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.1,
    "grid.color": "#CCCCCC",
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ======================= Helpers =======================

def color_for(model: str) -> str:
    return PALETTE.get(model, PALETTE["Default"])

def linestyle_for(model: str):
    return LINESTYLE.get(model, LINESTYLE["Default"])

def disp_name(model: str) -> str:
    return DISPLAY_MAP.get(model, model)

def stars_for_p(p: float | None) -> str:
    """映射 p 值到星号（***, **, *, ''）。"""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""

def format_axis_label(label: str) -> str:
    """把 CSV 的列名转成人类可读的标签。"""
    rep = {
        "_test": "",
        "_mm": " (mm)",
        "Balanced_Accuracy": "Balanced Accuracy",
        "F1_NLOS": "F1-Score",
        "Precision_NLOS": "Precision",
        "Recall_NLOS": "Recall",
        "RMSE": "RMSE",
        "MAE": "MAE",
    }
    for old, new in rep.items():
        label = label.replace(old, new)
    return label

def pointwise_ci(mean_y: np.ndarray, ys_stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """基于 per-seed 曲线栈计算 t-based 95% CI。"""
    n = ys_stack.shape[0]
    if n <= 1:
        return mean_y, mean_y
    sd = ys_stack.std(axis=0, ddof=1)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    rad = tcrit * sd / np.sqrt(n)
    lo = np.clip(mean_y - rad, 0, 1)
    hi = np.clip(mean_y + rad, 0, 1)
    return lo, hi

def find_cols(df: pd.DataFrame, base_names: List[str]):
    """
    返回 (mean_cols, ci_cols) 映射，优先使用 *_ci95；兼容旧版 *_ci95_radius。
    mean_cols: {metric_base -> metric_base_mean}
    ci_cols:   {metric_base -> metric_base_ci}
    """
    mean_cols, ci_cols = {}, {}
    for b in base_names:
        if f"{b}_mean" in df.columns:
            mean_cols[b] = f"{b}_mean"
        if f"{b}_ci95" in df.columns:
            ci_cols[b] = f"{b}_ci95"
        elif f"{b}_ci95_radius" in df.columns:
            ci_cols[b] = f"{b}_ci95_radius"
    return mean_cols, ci_cols

def lookup_pvalue(df: pd.DataFrame, model_name: str, main_model_key: str, metric_base: str) -> float | None:
    """
    优先返回 Holm 校正，其次 FDR，再退回原始 p；兼容旧版列名。
    """
    candidates = [
        f"p_holm({metric_base}) vs {main_model_key}",
        f"p_fdr({metric_base}) vs {main_model_key}",
        f"p({metric_base}) vs {main_model_key}",
    ]
    for col in candidates:
        if col in df.columns and model_name in df.index:
            try:
                return float(df.loc[model_name, col])
            except Exception:
                return None
    # legacy names
    legacy = [
        f"p-value (vs {main_model_key}) on {metric_base}",
        f"p-value vs {main_model_key} ({metric_base.replace('_test','')})",
    ]
    for col in legacy:
        if col in df.columns and model_name in df.index:
            try:
                return float(df.loc[model_name, col])
            except Exception:
                return None
    return None

def mat_key_for(model_key: str) -> str | None:
    """
    提供从 config 文件读取 .mat 文件名所需的变量名映射。
    这些变量应在 config_v7.py 或 config.py 中定义。
    """
    mapping = {
        "DualTransformer": "DUAL_TRANSFORMER_FINAL_RESULTS_MAT",
        "DNN": "DNN_FINAL_RESULTS_MAT",
        "SingleTransformer": "SINGLE_TRANSFORMER_FINAL_RESULTS_MAT",
        "SVM": "SVC_RIDGE_FINAL_RESULTS_MAT",
        "XGBoost": "XGBOOST_FINAL_RESULTS_MAT",
        "CNNLSTM": "CNN_LSTM_FINAL_RESULTS_MAT",
    }
    return mapping.get(model_key)

def discover_run_dirs() -> List[Path]:
    return sorted([d for d in BASE_RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_seed_")])

def seed_from_dirname(d: Path) -> int | None:
    m = re.search(r"run_seed_(\d+)", d.name)
    return int(m.group(1)) if m else None

def interp_curve(x: np.ndarray, y: np.ndarray, base_x: np.ndarray) -> np.ndarray:
    """对 ROC/PR 曲线做单调插值到统一 x 网格。"""
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    return np.interp(base_x, x_unique, y_unique)

# ======================= Load per-seed MAT results =======================

def load_all_runs_from_mat(seeds: List[int] | None = None, align_intersection: bool = True):
    """
    返回 dict[seed][model_key] -> dict{y_true, y_prob, errors_mm}
    - y_true: NLoS 标注（0/1）
    - y_prob: NLoS 预测概率
    - errors_mm: 预测误差（true - pred），仅用于误差 CDF
    """
    try:
        import config_v7 as config
    except ImportError:
        try:
            import config  # fallback
        except ImportError:
            raise SystemExit("❌ 找不到 config_v7.py 或 config.py，请将其放在脚本目录。")

    if seeds is None:
        run_dirs = discover_run_dirs()
        seeds = [seed_from_dirname(d) for d in run_dirs]
        run_map = {seed_from_dirname(d): d for d in run_dirs}
    else:
        run_map = {s: BASE_RESULTS_DIR / f"run_seed_{s}" for s in seeds}

    all_runs: Dict[int, Dict[str, Dict[str, np.ndarray | None]]] = {}
    ok_seeds: List[int] = []

    for sd, d in run_map.items():
        if sd is None:
            continue
        if not d.exists():
            print(f"⚠️ 跳过 seed {sd}: 目录不存在 → {d}")
            continue
        all_runs[sd] = {}
        for mk in MODEL_KEYS_FOR_MAT:
            mat_var = mat_key_for(mk)
            if not mat_var:
                continue
            mat_name = getattr(config, mat_var, None)
            if not mat_name:
                print(f"⚠️ config 缺少变量 {mat_var}，跳过 {mk}")
                all_runs[sd][mk] = {"y_true": None, "y_prob": None, "errors_mm": None}
                continue
            fp = d / mat_name
            if not fp.exists():
                all_runs[sd][mk] = {"y_true": None, "y_prob": None, "errors_mm": None}
                continue
            try:
                m = loadmat(fp)
                y_true = m.get("y_test_nlos", np.array([])).flatten()
                y_prob = m.get("y_pred_nlos_prob", np.array([])).flatten()
                if y_prob.size:
                    y_prob = np.where(y_prob == -999.0, np.nan, y_prob)
                true_err_mm = m.get("y_test_error_mm", np.array([])).flatten()
                pred_err_mm = m.get("y_pred_error_mm", np.array([])).flatten()
                mask = (~np.isnan(true_err_mm)) & (~np.isnan(pred_err_mm)) & (pred_err_mm != -999.0)
                errors_mm = (true_err_mm[mask] - pred_err_mm[mask]) if mask.any() else np.array([])
                all_runs[sd][mk] = {
                    "y_true": y_true if y_true.size else None,
                    "y_prob": y_prob if y_prob.size else None,
                    "errors_mm": errors_mm if errors_mm.size else None,
                }
            except Exception as e:
                print(f"  [错误] 读取 {fp.name} 失败: {e}")
                all_runs[sd][mk] = {"y_true": None, "y_prob": None, "errors_mm": None}
        ok_seeds.append(sd)

    # 打印 ROC/PR 可用的种子覆盖
    coverage = {mk: sorted([sd for sd in ok_seeds if all_runs.get(sd, {}).get(mk, {}).get("y_prob") is not None]) for mk in MODEL_KEYS_FOR_MAT}
    print("\n📋 种子覆盖情况 (ROC/PR 可用):")
    for mk in MODEL_KEYS_FOR_MAT:
        print(f"  - {disp_name(mk)}: seeds {coverage[mk]}")

    # 取交集，保证可比性
    if align_intersection:
        common = set(ok_seeds)
        for mk in MODEL_KEYS_FOR_MAT:
            common &= set(coverage[mk])
        common = sorted(common)
        if not common:
            print("⚠️ 没有所有模型的公共种子；按各自可用种子绘图（n 可能不同）。")
        else:
            drop = set(ok_seeds) - set(common)
            if drop:
                print(f"ℹ️ 仅使用所有模型共同的种子: {sorted(common)} (忽略 {sorted(drop)})")
            all_runs = {sd: all_runs[sd] for sd in common}

    return all_runs

# ======================= Curves: ROC / PR =======================

def plot_curves_with_ci(all_runs: Dict[int, Dict[str, Dict]], curve: str = "roc") -> None:
    """绘制 ROC 或 PR 曲线（per-seed 平均 + 95% CI）。"""
    if not all_runs:
        return

    # 找一个有 y_true 的样本（用于 PR baseline 的正类率）
    sample_y_true = None
    for run in all_runs.values():
        for mk in MODEL_KEYS_FOR_MAT:
            y_true = run.get(mk, {}).get("y_true")
            if y_true is not None:
                sample_y_true = y_true
                break
        if sample_y_true is not None:
            break
    if sample_y_true is None:
        print("⚠️ 无 y_true，跳过 ROC/PR。")
        return

    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    base_x = np.linspace(0, 1, 801)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    per_model = []

    for mk in MODEL_KEYS_FOR_MAT:
        ys, per_seed_metric = [], []
        for sd, run in all_runs.items():
            d = run.get(mk, {})
            y_true, y_prob = d.get("y_true"), d.get("y_prob")
            if y_true is None or y_prob is None or np.all(np.isnan(y_prob)):
                continue
            try:
                if curve == "roc":
                    x, y, _ = roc_curve(y_true, y_prob)
                    per_seed_metric.append(auc(x, y))
                else:
                    p, r, _ = precision_recall_curve(y_true, y_prob)
                    x, y = r, p  # PR: x=Recall, y=Precision
                    per_seed_metric.append(average_precision_score(y_true, y_prob))
                ys.append(interp_curve(x, y, base_x))
            except Exception:
                continue
        if not ys:
            continue
        ys = np.vstack(ys)
        mean_y = ys.mean(axis=0)
        lo, hi = pointwise_ci(mean_y, ys)
        per_model.append((mk, mean_y, lo, hi, float(np.nanmean(per_seed_metric)), ys.shape[0]))

    # 按 AUC / AP 排序（降序）
    per_model.sort(key=lambda t: t[4], reverse=True)

    for mk, mean_y, lo, hi, _, n in per_model:
        ax.plot(base_x, mean_y, color=color_for(mk), linestyle=linestyle_for(mk), lw=2.2, label=disp_name(mk), zorder=3)
        if n >= 2:
            ax.fill_between(base_x, lo, hi, facecolor=color_for(mk), alpha=0.2, zorder=2, edgecolor=color_for(mk), linewidth=0.8)

    if curve == "roc":
        ax.plot([0, 1], [0, 1], color="#808080", linestyle=":", lw=1.2, label="Chance (AUC = 0.5)", zorder=1)
        ax.set_xlabel("False Positive Rate", fontsize=11, fontweight='medium')
        ax.set_ylabel("True Positive Rate", fontsize=11, fontweight='medium')
        fn = "comparison_roc_curves_with_ci"
    else:
        pos_rate = np.mean(sample_y_true)
        ax.axhline(pos_rate, color="#808080", linestyle=":", lw=1.2, label="Random", zorder=1)
        ax.set_xlabel("Recall", fontsize=11, fontweight='medium')
        ax.set_ylabel("Precision", fontsize=11, fontweight='medium')
        fn = "comparison_pr_curves_with_ci"

    ax.set_xlim(-0.01, 1.01)
    if curve == "pr":
        ax.set_ylim(0.40, 1.02)                 # y 轴从 0.2 起
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    else:  # ROC
        ax.set_ylim(-0.01, 1.02)

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(handles, labels, loc="best", ncol=1, frameon=True, framealpha=0.95,
                        borderpad=0.3, columnspacing=1.0, handlelength=2.5, fontsize=10)
        leg.get_frame().set_linewidth(0.8)

    if per_model:
        n_seeds = per_model[0][5]
        #  ax.text(0.01, 0.03, f"Shaded: 95% CI across {n_seeds} seeds", transform=ax.transAxes,
        #         fontsize=9, color="#666666", ha='left', va='bottom', zorder=10)

    fig.tight_layout()
    plt.savefig(PLOTS_DIR / f"{fn}.png", format="png")
    plt.close(fig)
    print(f"   ✓ Saved: {fn}.png")

# ======================= Error CDF =======================

def plot_error_cdf_with_ci(all_runs: Dict[int, Dict[str, Dict]]) -> None:
    """绘制绝对误差 CDF（per-seed 平均 + 95% CI）。"""
    if not all_runs:
        return

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    per_model = []

    for mk in MODEL_KEYS_FOR_MAT:
        series = []
        for sd, run in all_runs.items():
            e = run.get(mk, {}).get("errors_mm")
            if e is not None and len(e) > 1:
                series.append(e)
        if not series:
            continue
        all_err = np.concatenate(series)
        base_x = np.linspace(0, np.percentile(np.abs(all_err), 99.5), 400)

        curves = []
        for e in series:
            ae = np.sort(np.abs(e))
            y = np.arange(1, len(ae) + 1) / len(ae)
            curves.append(np.interp(base_x, ae, y, right=1.0))
        ys = np.vstack(curves)
        mean_y = ys.mean(axis=0)
        lo, hi = pointwise_ci(mean_y, ys)
        med_err = base_x[np.searchsorted(mean_y, 0.5)]
        per_model.append((mk, base_x, mean_y, lo, hi, ys.shape[0], med_err))

    # 小位移按中位误差排序（小的更优）
    per_model.sort(key=lambda t: t[6])

    for mk, base_x, mean_y, lo, hi, n, _ in per_model:
        ax.plot(base_x, mean_y, color=color_for(mk), linestyle=linestyle_for(mk), lw=2.2, label=disp_name(mk), zorder=3)
        if n >= 2:
            ax.fill_between(base_x, lo, hi, facecolor=color_for(mk), alpha=0.2, zorder=2, edgecolor=color_for(mk), linewidth=0.8)

    # 50% / 90% 辅助线
    ax.axhline(0.9, color="#CCCCCC", linestyle=":", lw=0.8, alpha=0.7, zorder=1)
    ax.axhline(0.5, color="#CCCCCC", linestyle=":", lw=0.8, alpha=0.7, zorder=1)
    xmax = ax.get_xlim()[1]
    ax.text(xmax * 0.98, 0.92, '90%', fontsize=8, color="#666666", ha='right')
    ax.text(xmax * 0.98, 0.52, '50%', fontsize=8, color="#666666", ha='right')

    ax.set_xlabel("Absolute Prediction Error (mm)", fontsize=11, fontweight='medium')
    ax.set_ylabel("Cumulative Distribution Function", fontsize=11, fontweight='medium')
    ax.set_ylim(-0.01, 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(handles, labels, loc="best", ncol=1, frameon=True, framealpha=0.95,
                        borderpad=0.3, columnspacing=1.0, handlelength=2.5, fontsize=10)
        leg.get_frame().set_linewidth(0.8)

    if per_model:
        n_seeds = per_model[0][5]
        # ax.text(0.01, 0.03, f"Shaded: 95% CI across {n_seeds} seeds", transform=ax.transAxes,
        #         fontsize=9, color="#666666", ha='left', va='bottom', zorder=10)

    fig.tight_layout()
    fn = "comparison_error_cdf_with_ci"
    plt.savefig(PLOTS_DIR / f"{fn}.png", format='png')
    plt.close(fig)
    print(f"   ✓ Saved: {fn}.png")

# ======================= Bars: means ± 95% CI and p-values =======================

def plot_bars_with_ci_and_pvalues(summary_csv: Path, main_model_key: str = MAIN_MODEL_KEY_FOR_CSV) -> None:
    """从 CSV 读取均值/CI/p 值绘制条形图组。"""
    if not summary_csv.exists():
        print(f"⚠️ 未找到统计汇总 CSV: {summary_csv}，跳过条形图。")
        return

    df = pd.read_csv(summary_csv, index_col="Model")

    plot_configs = [
        {
            "metrics": ["Balanced_Accuracy_test", "F1_NLOS_test", "Precision_NLOS_test", "Recall_NLOS_test"],
            "ylabel": "Score",
            "fn": "bar_chart_classification_four_metrics",
            "ylim": (0.7, 1.02),
            "is_classification": True,
            "add_sig": False,
        },
        {
            "metrics": ["RMSE_mm_test", "MAE_mm_test"],
            "ylabel": "Error (mm)",
            "fn": "bar_chart_regression",
            "ylim": None,
            "is_classification": False,
            "add_sig": True,  # 显著性星号仅对回归图
        },
    ]

    for pc in plot_configs:
        mcols, ccols = find_cols(df, pc["metrics"])
        if not mcols:
            continue
        dfm = df[list(mcols.values())].rename(columns={v: format_axis_label(k) for k, v in mcols.items()})
        dfc = df[list(ccols.values())] if ccols else pd.DataFrame(index=df.index)
        _plot_groupbar(
            df_mean=dfm, df_ci=dfc, mcols=mcols, ccols=ccols,
            ylabel=pc["ylabel"], fn=pc["fn"], add_sig=pc.get("add_sig", False),
            raw_df=df, main_model_key=main_model_key, ylim=pc.get("ylim"),
            is_classification=pc.get("is_classification", True),
        )

def _plot_groupbar(
    df_mean: pd.DataFrame,
    df_ci: pd.DataFrame,
    mcols: Dict[str, str],
    ccols: Dict[str, str],
    ylabel: str,
    fn: str,
    add_sig: bool,
    raw_df: pd.DataFrame,
    main_model_key: str,
    ylim: Tuple[float, float] | None,
    is_classification: bool = True,
) -> None:
    """绘制分组条形图并可选添加显著性星号。"""

    # 按 CSV 实际索引顺序（避免硬编码丢模型）
    idx = list(df_mean.index)
    disp_idx = [disp_name(i) for i in idx]
    x = np.arange(len(idx))
    num_metrics = df_mean.shape[1]

    # 颜色
    METRIC_COLORS = {
        "Balanced Accuracy": "#5B9BD5",
        "F1-Score": "#FFC000",
        "Precision": "#A5609C",
        "Recall": "#70AD47",
    }
    REGRESSION_COLORS = {"RMSE (mm)": "#E74C3C", "MAE (mm)": "#3498DB"}

    width = 0.8 / max(1, num_metrics)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    max_bar_top = -np.inf

    for j, col_display_name in enumerate(df_mean.columns):
        y = df_mean.loc[idx, col_display_name].astype(float).values

        # 找到对应的 metric base 名（用于找 CI 和 p 值 列）
        metric_base = next((k for k, v in mcols.items() if format_axis_label(k) == col_display_name), None)

        # CI 半径
        if metric_base and metric_base in ccols:
            yerr_col = ccols[metric_base]
            yerr = raw_df.loc[idx, yerr_col].astype(float).values if yerr_col in raw_df.columns else None
        else:
            yerr = None

        offset = j * width - (num_metrics - 1) * width / 2

        # 颜色
        if is_classification:
            bar_color = METRIC_COLORS.get(col_display_name, "#808080")
            bars = ax.bar(x + offset, y, width=width, label=col_display_name,
                          color=bar_color, edgecolor="#333333", linewidth=0.8, zorder=2)
        else:
            bar_color = REGRESSION_COLORS.get(col_display_name, "#95A5A6")
            bars = ax.bar(x + offset, y, width=width, label=col_display_name,
                          color=bar_color, edgecolor="#2C3E50", linewidth=0.8, alpha=0.88, zorder=2)

        if yerr is not None:
            ax.errorbar(x + offset, y, yerr=yerr, fmt="none",
                        ecolor="#333333", elinewidth=1.0, capsize=3, capthick=1.0, zorder=3)

        tops = y + (np.nan_to_num(yerr, nan=0.0) if yerr is not None else 0.0)
        if len(tops):
            max_bar_top = max(max_bar_top, np.nanmax(tops))

        # 显著性星号（仅回归图）
        if add_sig and metric_base is not None:
            for i, model_name in enumerate(idx):
                if model_name == main_model_key:
                    continue
                p = lookup_pvalue(raw_df, model_name, main_model_key, metric_base)
                s = stars_for_p(p)
                if s:
                    y_top = y[i] + (yerr[i] if (yerr is not None and i < len(yerr) and not np.isnan(yerr[i])) else 0.0)
                    # 适度抬高一点，避免覆盖误差棒
                    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
                    ax.text(bars[i].get_x() + bars[i].get_width() / 2.0,
                            y_top + y_offset,
                            s, ha="center", va="bottom", fontsize=11, color="#C0392B", fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(disp_idx, fontsize=11, fontweight='medium')
    ax.set_xlabel("Model", fontsize=12, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='medium')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if is_classification:
            fig.subplots_adjust(top=0.88)
            pretty = ["Balanced Acc" if l == "Balanced Accuracy" else ("F1" if l == "F1-Score" else l) for l in labels]
            leg = fig.legend(handles, pretty, loc='upper center', bbox_to_anchor=(0.5, 0.975), ncol=4,
                             frameon=True, fancybox=False, framealpha=0.97,
                             borderpad=0.2, columnspacing=1.5, handlelength=1.8, fontsize=14)
            leg.get_frame().set_linewidth(0.8)
            leg.get_frame().set_edgecolor('#CCCCCC')
        else:
            leg = ax.legend(handles, labels, loc='upper right', frameon=True, fancybox=False, framealpha=0.95,
                            borderpad=0.4, columnspacing=1.0, handlelength=1.8, fontsize=14)
            leg.get_frame().set_linewidth(1.0)
            leg.get_frame().set_edgecolor('#34495E')

    ax.set_ylim(bottom=0)

    # 网格线
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.5, color='#808080')

    # Y 轴刻度
    if ylim and isinstance(ylim, tuple) and len(ylim) == 2:
        ax.set_ylim(*ylim)
        if is_classification and ylim[1] <= 1.2:
            # 主刻度：0.60、0.70、0.80、0.90、1.00
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            # 次刻度：每 0.05
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            # 显示为 60、70、...、100（不带 % 符号，如需带百分号把 {x*100:.0f}% 即可）
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*100:.0f}"))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))

    fig.tight_layout()
    out_path = PLOTS_DIR / f"{fn}.png"
    plt.savefig(out_path, format="png")
    plt.close(fig)
    print(f"   ✓ Saved: {fn}.png")

# ======================= Main =======================

def main():
    print("\n" + "=" * 68)
    print("  0_plot_results — Publication-Ready Plotting (complete version)")
    print("=" * 68 + "\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='*', type=int, help='Specify seeds to include, e.g., --seeds 42 43 44')
    parser.add_argument('--no-align', action='store_true', help='Do NOT restrict to intersection of seeds across models (curves only)')
    args = parser.parse_args()

    if not BASE_RESULTS_DIR.exists():
        raise SystemExit(f"❌ 结果目录不存在：{BASE_RESULTS_DIR}")

    # Bars (needs CSV)
    if SUMMARY_CSV.exists():
        plot_bars_with_ci_and_pvalues(SUMMARY_CSV, main_model_key=MAIN_MODEL_KEY_FOR_CSV)
    else:
        print(f"⚠️ 未找到 {SUMMARY_CSV.name}，跳过条形图。")

    # Curves (needs MATs)
    all_runs = load_all_runs_from_mat(seeds=args.seeds, align_intersection=(not args.no_align))
    if all_runs:
        print("\n📈 绘制带 95% CI 的 ROC/PR/CDF 曲线...")
        plot_curves_with_ci(all_runs, curve="roc")
        plot_curves_with_ci(all_runs, curve="pr")
        plot_error_cdf_with_ci(all_runs)
    else:
        print("⚠️ 未能载入任何 MAT 结果，跳过 ROC/PR/CDF。")

    print("\n" + "=" * 68)
    print("✨ 图表生成完成！")
    print(f"📁 输出目录: {PLOTS_DIR.resolve()}")

if __name__ == "__main__":
    main()