# -*- coding: utf-8 -*-
"""
最终统计分析脚本 (v3.0)
- 自动发现 run_seed_* 下的汇总 CSV，合并后按模型统计均值与95%CI
- 与主模型进行“单侧配对 t 检验”（回归指标更小更好；分类指标更大更好）
- 输出：p 值(科学计数法)、Holm/FDR 校正后的 p 值、配对 Cohen's d_z、均值差及其95%CI
- 终端打印与CSV均保留科学计数法，不再把极小 p 值四舍五入为 0
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# 尝试加载多重比较修正
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# ====================== 配置 ======================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_RESULTS_DIR = SCRIPT_DIR / "Ghent_Statistical_Runs"   # 确保该目录存在
SUMMARY_FILENAME_PATTERN = "all_models_summary_combined_*fold_cv*.csv"

MAIN_MODEL = "DualTransformer"  # 若你的主模型在CSV里叫“DCUT”，这里改成 "DCUT"

# 指标列名（需与汇总CSV一致）
METRICS_TO_ANALYZE = [
    "Balanced_Accuracy_test",
    "F1_NLOS_test",
    "Precision_NLOS_test",
    "Recall_NLOS_test",
    "RMSE_mm_test",
    "MAE_mm_test",
]

# 哪些指标“越小越好 / 越大越好”
BETTER_SMALLER = {"RMSE_mm_test", "MAE_mm_test"}
BETTER_LARGER  = {"Balanced_Accuracy_test", "F1_NLOS_test",
                  "Precision_NLOS_test", "Recall_NLOS_test"}

# ================ 工具函数 =================
def ci95_radius(values: np.ndarray) -> float:
    """t 分布下 95% 置信区间半径（基于样本均值）"""
    values = np.asarray(values, float)
    n = np.sum(~np.isnan(values))
    if n < 2:
        return np.nan
    sd = np.nanstd(values, ddof=1)
    tcrit = stats.t.ppf(0.975, n - 1)
    return float(tcrit * sd / np.sqrt(n))

def paired_stats(main_vals: np.ndarray, base_vals: np.ndarray, metric: str):
    """
    单侧配对t检验 + 配对效应量 + 差值95%CI
    对于 BETTER_SMALLER：H1: main < base（alternative='less'），差值 diff = main - base 期望为负
    对于 BETTER_LARGER：H1: main > base（alternative='greater'），差值 diff = main - base 期望为正
    返回：p, p_alt, mean_diff, (ci_lo, ci_hi), dz
    """
    assert len(main_vals) == len(base_vals) and len(main_vals) >= 2
    diff = main_vals - base_vals
    n = len(diff)
    sd_diff = np.std(diff, ddof=1)
    mean_diff = float(np.mean(diff))
    # 配对 Cohen's d_z
    dz = float(mean_diff / sd_diff) if sd_diff > 0 else np.nan
    # 95% CI for mean difference
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (mean_diff - tcrit * sd_diff / np.sqrt(n),
          mean_diff + tcrit * sd_diff / np.sqrt(n))

    # 单侧检验方向
    if metric in BETTER_SMALLER:
        alt = "less"
    elif metric in BETTER_LARGER:
        alt = "greater"
    else:
        alt = "two-sided"

    t_res = stats.ttest_rel(main_vals, base_vals, alternative=alt)
    p = float(t_res.pvalue)
    return p, alt, mean_diff, ci, dz

def sci(p: float, min_text: str = "<1e-12") -> str:
    """p值格式化为科学计数法；极小值给阈值文本"""
    if pd.isna(p):
        return ""
    if p < 1e-12:
        return min_text
    return f"{p:.3e}"

# ================ 主流程 =================
def analyze_and_summarize():
    if not BASE_RESULTS_DIR.exists():
        raise SystemExit(f"目录不存在：{BASE_RESULTS_DIR}")

    run_dirs = sorted(d for d in BASE_RESULTS_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("run_seed_"))
    if not run_dirs:
        raise SystemExit(f"未找到任何 run_seed_* 目录：{BASE_RESULTS_DIR}")

    all_run_rows = []
    for rd in run_dirs:
        csvs = list(rd.glob(SUMMARY_FILENAME_PATTERN))
        if not csvs:
            print(f"[跳过] {rd.name}: 未找到 {SUMMARY_FILENAME_PATTERN}")
            continue
        if len(csvs) > 1:
            print(f"[提示] {rd.name}: 找到 {len(csvs)} 个摘要文件，使用 {csvs[0].name}")
        df = pd.read_csv(csvs[0])
        keep = ["Model"] + [c for c in METRICS_TO_ANALYZE if c in df.columns]
        df = df[keep].copy()
        df["run_id"] = rd.name
        all_run_rows.append(df)

    if not all_run_rows:
        raise SystemExit("未收集到任何结果数据。")

    combined = pd.concat(all_run_rows, ignore_index=True)
    models = combined["Model"].unique().tolist()
    print(f"检测到模型：{', '.join(models)}")
    if MAIN_MODEL not in models:
        print(f"[警告] 未找到主模型 {MAIN_MODEL}，将只做均值与CI，不做配对检验。")

    # 1) 每模型的均值与95%CI
    # 1) 每模型的均值与95%CI
    rows = []
    for model, g in combined.groupby("Model"):
        row = {"Model": model, "Runs": g["run_id"].nunique()}
        for m in METRICS_TO_ANALYZE:
            if m not in g:
                continue
            vals = g[m].astype(float).to_numpy()

            # 均值
            mean = np.nanmean(vals)
            row[f"{m}_mean"] = mean

            # 95%CI 半径（t 分布），以及可选的上下界，便于核查
            ci = ci95_radius(vals)         # ← 关键：输出 *_ci95_radius
            row[f"{m}_ci95_radius"] = ci
            if not np.isnan(ci):
                row[f"{m}_ci95_lower"] = mean - ci
                row[f"{m}_ci95_upper"] = mean + ci
            else:
                row[f"{m}_ci95_lower"] = np.nan
                row[f"{m}_ci95_upper"] = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("Model").sort_index()


    # 2) 与主模型的单侧配对 t 检验 + 效应量
    if MAIN_MODEL in models:
        main = combined[combined["Model"] == MAIN_MODEL]
        for base_model in models:
            if base_model == MAIN_MODEL:
                continue
            base = combined[combined["Model"] == base_model]
            # 以 run_id 对齐，确保配对
            merged = pd.merge(
                main[["run_id"] + METRICS_TO_ANALYZE],
                base[["run_id"] + METRICS_TO_ANALYZE],
                on="run_id",
                suffixes=("_main", "_base")
            ).dropna()

            for metric in METRICS_TO_ANALYZE:
                mcol = f"{metric}_main"
                bcol = f"{metric}_base"
                vals = merged[[mcol, bcol]].dropna()
                if len(vals) < 2:
                    summary.loc[base_model, f"p({metric}) vs {MAIN_MODEL}"] = np.nan
                    summary.loc[base_model, f"alt({metric})"] = ""
                    summary.loc[base_model, f"dz({metric})"] = np.nan
                    summary.loc[base_model, f"diff({metric})"] = np.nan
                    summary.loc[base_model, f"diff({metric})_ci_lo"] = np.nan
                    summary.loc[base_model, f"diff({metric})_ci_hi"] = np.nan
                    continue

                p, alt, mean_diff, ci, dz = paired_stats(
                    vals[mcol].to_numpy(float),
                    vals[bcol].to_numpy(float),
                    metric
                )
                summary.loc[base_model, f"p({metric}) vs {MAIN_MODEL}"] = p
                summary.loc[base_model, f"alt({metric})"] = alt
                summary.loc[base_model, f"dz({metric})"] = dz
                summary.loc[base_model, f"diff({metric})"] = mean_diff
                summary.loc[base_model, f"diff({metric})_ci_lo"] = ci[0]
                summary.loc[base_model, f"diff({metric})_ci_hi"] = ci[1]

        # 3) 多重比较修正（Holm 与 FDR）
        if HAS_STATSMODELS:
            pcols = [c for c in summary.columns if c.startswith("p(") and f"vs {MAIN_MODEL}" in c]
            if pcols:
                pvals = summary[pcols].to_numpy().ravel()
                mask = ~pd.isna(pvals)
                if mask.any():
                    # Holm
                    _, p_adj_holm, _, _ = multipletests(pvals[mask], alpha=0.05, method="holm")
                    # FDR-BH
                    _, p_adj_bh, _, _ = multipletests(pvals[mask], alpha=0.05, method="fdr_bh")
                    padj_holm = np.full_like(pvals, np.nan, dtype=float); padj_holm[mask] = p_adj_holm
                    padj_bh   = np.full_like(pvals, np.nan, dtype=float); padj_bh[mask]   = p_adj_bh
                    holm_cols = [c.replace("p(", "p_holm(") for c in pcols]
                    bh_cols   = [c.replace("p(", "p_fdr(")  for c in pcols]
                    summary[holm_cols] = pd.DataFrame(padj_holm.reshape(summary[pcols].shape),
                                                      index=summary.index, columns=holm_cols)
                    summary[bh_cols]   = pd.DataFrame(padj_bh.reshape(summary[pcols].shape),
                                                      index=summary.index, columns=bh_cols)

    # 4) 友好打印（科学计数法）
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n===== 模型均值 ± 95%CI =====")
    to_show = summary.copy()
    # 把 p 值列格式化为字符串（科学计数法）
    for c in to_show.columns:
        if c.startswith("p(") or c.startswith("p_holm(") or c.startswith("p_fdr("):
            to_show[c] = to_show[c].apply(sci)
    print(to_show)

    # 5) 保存CSV（不使用 float_format，保留科学计数法）
    out_csv = BASE_RESULTS_DIR / "final_statistical_summary_with_pvalues.csv"
    summary.to_csv(out_csv, index=True)
    print(f"\n✅ 已保存：{out_csv}")

if __name__ == "__main__":
    analyze_and_summarize()
