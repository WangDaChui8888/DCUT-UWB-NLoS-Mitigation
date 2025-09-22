# -*- coding: utf-8 -*-
"""
Enhanced BA aggregator with 95% CI (self-rooted) + SCI LaTeX table
==================================================================
运行位置：随意。脚本会以**脚本文件所在路径**为根。
目录约定（相对脚本路径）：
  Tuning_Baselines_R1/
    CNNLSTM/run_*/all_models_summary_combined_3fold_cv_v7.csv
    DNN/run_*/...
    LS-SVM/run_*/...
    SingleTransformer/run_*/...
    XGBoost/run_*/...

输出（相对脚本路径）：
  Ghent_Statistical_Runs/
    ba_95ci_summary.csv        # 折 × run 总体统计（mean, std, n, 95%CI）
    ba_best_runs.csv           # 各算法最佳 run + 其95%CI + 超参
    ba_best_runs_table.tex     # SCI风格LaTeX表格（最佳BA+95%CI+超参组合）

用法：
  python aggregate_ba_ci_autoroot_best_ci.py
  # 可选：指定需要聚合的算法名
  python aggregate_ba_ci_autoroot_best_ci.py --algos CNNLSTM DNN LS-SVM SingleTransformer XGBoost
  # 可选：不生成LaTeX表
  python aggregate_ba_ci_autoroot_best_ci.py --no-latex
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# 默认 5 种算法（如你要加 DCUT，请在 --algos 中额外传入 DCUT 并在目录下放对应结构）
ALGOS_DEFAULT = ["CNNLSTM", "DNN", "LS-SVM", "SingleTransformer", "XGBoost"]
TARGET_NAME = "all_models_summary_combined_3fold_cv_v7.csv"

# 为LaTeX显示友好，把名称做统一
def norm_method_name(m: str) -> str:
    return (m.replace("SingleTransformer", "ST")
             .replace("CNNLSTM", "CNN-LSTM"))

# 允许展示的（拍平后）超参键清单
ALLOW_KEYS = {
    "ST": {"transformer_layers","attention_heads","ff_dim",
           "loss_weight_error","learning_rate","dropout_rate"},
    "DNN": {"dense_units","dropout_rate","learning_rate",
            "weight_decay","activation","loss_weight_error"},
    "LS-SVM": {"svm_kernel","svm_C","svm_gamma","svm_class_weight","standardize"},
    "XGBoost": {"xgb_n_estimators","xgb_max_depth","xgb_learning_rate",
                "xgb_subsample","xgb_colsample_bytree","xgb_gamma","xgb_min_child_weight"},
    "CNN-LSTM": {"conv_filters","kernel_size","pool_size","lstm_units",
                 "dense_units","dropout_rate","learning_rate","activation","loss_weight_error"},
}

def find_run_csvs(algo_dir: Path) -> List[Path]:
    """找到某算法的所有 run_* 里的目标CSV"""
    if not algo_dir.exists():
        return []
    csvs = []
    for p in sorted(algo_dir.glob("run_*")):
        f = p / TARGET_NAME
        if f.exists():
            csvs.append(f)
    return csvs

def extract_bal_acc_and_cfg(df: pd.DataFrame) -> Tuple[List[float], List[str]]:
    """
    返回：该CSV中的 BA 列表（通常是每折一行的 bal_acc），以及 best_cfg 的字符串列表（如有）
    兼容：
      - per-fold: 有 'bal_acc' 列
      - 单行汇总: 有 'balanced_accuracy_cv_mean'（此时只能返回一个数；cfg尽力找）
    """
    ba_vals: List[float] = []
    cfgs: List[str] = []

    if "bal_acc" in df.columns:
        ba_vals = df["bal_acc"].astype(float).tolist()
        if "best_cfg" in df.columns:
            cfgs = [str(x) for x in df["best_cfg"].tolist()]
    elif "balanced_accuracy_cv_mean" in df.columns:
        ba_vals = [float(df.loc[0, "balanced_accuracy_cv_mean"])]
        # 试探任何可能描述配置的列
        for c in df.columns:
            cl = c.lower()
            if "cfg" in cl or "param" in cl:
                try:
                    cfgs = [str(df.loc[0, c])]
                    break
                except Exception:
                    pass
    else:
        # fuzzy: 找名字里带 balanced / acc 且 mean/val 的列
        for c in df.columns:
            cl = c.lower()
            if "balanced" in cl and "acc" in cl and ("mean" in cl or "val" in cl):
                try:
                    ba_vals = [float(df.loc[0, c])]
                    break
                except Exception:
                    pass

    return ba_vals, cfgs

def ci95(values: List[float]) -> Tuple[float,float,int,float,float,float]:
    """t分布 95%CI：返回 mean, std, n, lower, upper, half"""
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(np.mean(x)) if n > 0 else float("nan")
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n > 1:
        tval = stats.t.ppf(0.975, df=n-1)
        half = float(tval * std / np.sqrt(n))
    else:
        half = 0.0
    return mean, std, n, mean - half, mean + half, half

def most_common_cfg(cfg_list: List[str]) -> str:
    """挑选出现次数最多的 cfg 字符串（若是JSON，转为键排序的标准串再投票）"""
    cfg_list = [c for c in cfg_list if isinstance(c, str) and c.strip()]
    if not cfg_list:
        return ""
    normed = []
    for c in cfg_list:
        s = str(c).strip()
        try:
            obj = json.loads(s)
            normed.append(json.dumps(obj, sort_keys=True))
        except Exception:
            normed.append(s)
    # 投票
    counts = {}
    for s in normed:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]

def parse_cfg_pairs(raw: str, method_display: str) -> List[Tuple[str, str]]:
    """
    把 best_cfg_json（可能是json或'--k v'/'k=v'串）解析为 (key, value) 对，并按 ALLOW_KEYS 过滤。
    """
    if not isinstance(raw, str) or not raw.strip():
        return []
    s = raw.strip()
    # 优先尝试 JSON
    items = []
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            items = list(obj.items())
        elif isinstance(obj, list):
            for o in obj:
                if isinstance(o, dict):
                    items.extend(o.items())
                else:
                    items.append(("cfg", str(o)))
        else:
            items = [("cfg", str(obj))]
    except Exception:
        # 非JSON：尝试解析 --key value 或 key=value
        tokens = s.replace(",", " ").split()
        i = 0
        tmp = []
        while i < len(tokens):
            t = tokens[i]
            if t.startswith("--"):
                key = t.lstrip("-")
                val = tokens[i+1] if i+1 < len(tokens) else ""
                tmp.append((key, val)); i += 2
            elif "=" in t:
                key, val = t.split("=", 1)
                tmp.append((key.strip("-"), val)); i += 1
            else:
                i += 1
        items = tmp

    allow = ALLOW_KEYS.get(method_display, None)
    out = []
    seen = set()
    for k, v in items:
        k = str(k).strip().lstrip("-")
        v = str(v).strip()
        if v == "":
            continue
        if (allow is None) or (k in allow):
            pair = (k, v)
            if pair not in seen:
                out.append(pair); seen.add(pair)
    return out[:12]  # 最大12项，避免列过长

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="*", default=ALGOS_DEFAULT,
                    help="要聚合的算法文件夹名（默认5种）")
    ap.add_argument("--no-latex", action="store_true", help="不生成 LaTeX 表")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir / "Tuning_Baselines_R1"
    out_dir = script_dir / "Ghent_Statistical_Runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled_rows = []
    best_rows = []

    for algo in args.algos:
        algo_dir = root / algo
        run_csvs = find_run_csvs(algo_dir)
        if not run_csvs:
            print(f"[WARN] 未找到 {algo_dir} 下的 run_* CSV")
            continue

        # ---- pooled across runs × folds ----
        pooled_vals: List[float] = []
        for csv in run_csvs:
            df = pd.read_csv(csv)
            ba_vals, _ = extract_bal_acc_and_cfg(df)
            pooled_vals.extend(ba_vals)
        if pooled_vals:
            mean, std, n, lo, hi, half = ci95(pooled_vals)
            pooled_rows.append({
                "Method": norm_method_name(algo),
                "BA_mean": mean, "BA_std": std, "n": n,
                "CI95_lower": lo, "CI95_upper": hi, "CI95_half": half
            })

        # ---- best run (by mean BA across its folds) ----
        per_run = []
        for csv in run_csvs:
            df = pd.read_csv(csv)
            ba_vals, cfgs = extract_bal_acc_and_cfg(df)
            if not ba_vals:
                continue
            mean, std, n, lo, hi, half = ci95(ba_vals)
            per_run.append({
                "run": csv.parent.name,
                "mean": mean, "std": std, "n_folds": n,
                "CI95_lower": lo, "CI95_upper": hi, "CI95_half": half,
                "cfg_json": most_common_cfg(cfgs),
            })
        if per_run:
            per_run.sort(key=lambda d: d["mean"], reverse=True)
            best = per_run[0]
            best_rows.append({
                "Method": norm_method_name(algo),
                "best_run": best["run"],
                "BA_mean_cv": best["mean"],
                "BA_std_cv": best["std"],
                "n_folds": best["n_folds"],
                "CI95_lower": best["CI95_lower"],
                "CI95_upper": best["CI95_upper"],
                "CI95_half": best["CI95_half"],
                "best_cfg_json": best["cfg_json"],
            })

    # ---- 写出CSV ----
    def order_idx(m):
        order = ["DCUT", "ST", "CNN-LSTM", "DNN", "XGBoost", "LS-SVM"]
        return order.index(m) if m in order else 999

    if pooled_rows:
        df_pool = pd.DataFrame(pooled_rows).sort_values(by="Method", key=lambda s: s.map(order_idx))
        df_pool.to_csv(out_dir / "ba_95ci_summary.csv", index=False)
        print(f"[OK] 写出：{out_dir / 'ba_95ci_summary.csv'}")
    else:
        df_pool = None

    if best_rows:
        df_best = pd.DataFrame(best_rows).sort_values(by="Method", key=lambda s: s.map(order_idx))
        df_best.to_csv(out_dir / "ba_best_runs.csv", index=False)
        print(f"[OK] 写出：{out_dir / 'ba_best_runs.csv'}")
    else:
        df_best = None

    # ---- 生成 SCI 风格 LaTeX 表 ----
    if (not args.no_latex) and (df_best is not None):
        rows = []
        # 便于 LaTeX 多行配置的紧凑展示
        for _, r in df_best.iterrows():
            method = str(r["Method"])
            ba_ci = f"{r['BA_mean_cv']:.3f} [95\\% CI {r['CI95_lower']:.3f}, {r['CI95_upper']:.3f}]"
            best_run = str(r["best_run"])
            cfg_pairs = parse_cfg_pairs(str(r.get("best_cfg_json","")), method)
            if cfg_pairs:
                cfg_lines = [f"{k}={v}" for k,v in cfg_pairs]
                cfg_tex = "\\scriptsize\\begin{tabular}[t]{@{}l@{}}" + " \\\\ ".join(cfg_lines) + "\\end{tabular}"
            else:
                cfg_tex = "\\scriptsize n/a"
            pooled_str = "--"
            if df_pool is not None and not df_pool[df_pool["Method"]==method].empty:
                pm = float(df_pool[df_pool["Method"]==method].iloc[0]["BA_mean"])
                ps = float(df_pool[df_pool["Method"]==method].iloc[0]["BA_std"])
                pooled_str = f"{pm:.3f} $\\pm$ {ps:.3f}"
            rows.append((method, ba_ci, best_run, cfg_tex, pooled_str))

        rows.sort(key=lambda x: order_idx(x[0]))

        tex = []
        tex.append(r"\begin{table}[t]")
        tex.append(r"\centering")
        tex.append(r"\small")
        tex.append(r"\caption{Best validated configurations per baseline after systematic hyper-parameter tuning and early stopping. For each method, we report the \textbf{best run}'s Balanced Accuracy (BA) with 95\% confidence interval computed across outer folds, the run identifier, and the corresponding hyper-parameters. Pooled mean$\pm$std across all runs$\times$folds are shown for reference.}")
        tex.append(r"\label{tab:best_ba_configs}")
        tex.append(r"\setlength{\tabcolsep}{6pt}")
        tex.append(r"\begin{tabular}{l l l p{0.42\linewidth} l}")
        tex.append(r"\toprule")
        tex.append(r"\textbf{Method} & \textbf{Best BA [95\% CI]} & \textbf{Best Run} & \textbf{Hyper-parameter combination} & \textbf{Pooled BA (mean$\pm$std)} \\")
        tex.append(r"\midrule")
        for method, ba_ci, best_run, cfg_tex, pooled_str in rows:
            tex.append(f"{method} & {ba_ci} & {best_run} & {cfg_tex} & {pooled_str} \\\\")
        tex.append(r"\bottomrule")
        tex.append(r"\vspace{2pt}")
        tex.append(r"\footnotesize Notes: Best run is selected by highest fold-mean BA within each method under identical outer splits and random seeds across baselines. 95\% CIs use $t_{{0.975,\,n-1}}\,s/\sqrt{{n}}$ over the best run's outer-fold scores.}")
        tex.append(r"\end{tabular}")
        tex.append(r"\end{table}")
        (out_dir / "ba_best_runs_table.tex").write_text("\n".join(tex), encoding="utf-8")
        print(f"[OK] 写出：{out_dir / 'ba_best_runs_table.tex'}")

if __name__ == "__main__":
    main()
