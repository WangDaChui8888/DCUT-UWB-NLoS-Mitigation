# -*- coding: utf-8 -*-
"""
Systematic hyper-parameter tuning for baselines (DNN, ST, LS-SVM, XGBoost, CNN-LSTM)
- Unified early stopping protocol (val-BA) and logging.
- Writes a CSV of all trials and prints per-model best config.
"""
import subprocess
from pathlib import Path
import time
import pandas as pd
import itertools
from datetime import datetime
# ===================== 基本路径配置 =====================
MAIN_SCRIPT = "4_main_script_baselines.py"
PROJECT_DIR = Path(__file__).resolve().parent
BASE_SAVE_DIR = PROJECT_DIR / "Tuning_Baselines_R1"
LOG_FILE_PATH = PROJECT_DIR / "tuning_log_baselines.txt"

# 你的汇总文件名（从 config 中读取失败时的兜底）
DEFAULT_SUMMARY_CSV = "summary.csv"

# 结果CSV里“得分列”的候选名（按优先级尝试）
# 修改后
SCORE_COLUMN_CANDIDATES = [
    "balanced_accuracy_cv_mean",   # <-- 改成小写，与CSV文件中的列名完全一致
    "val_BA",                      
    "BA_cv_mean",                  
    "BA"                           
]

# ===================== 顺序调参建议（可选阅读） =====================
# 你已经在主体论文对 DCUT(我们的模型)做了顺序调参（Architecture→Learning→Capacity）
# 这里针对“基线”做一次“并列的系统化搜索”，每个模型给出一个搜索空间和固定训练协议。

# ===================== 公共训练/早停参数（深度模型） =====================
# 若 main_script 支持这些 flag，则会生效；不支持也无妨（多余 flag 会被忽略或需你改名）
COMMON_DEEP_ARGS = [
    "--max_epochs", "20",                 # 统一训练预算
    "--batch_size", "128",
    "--optimizer", "adam",
    "--early_stopping", "true",
    "--es_monitor", "val_BA",             # 监控验证集 BA
    "--es_patience", "8",
    "--es_min_delta", "1e-4",
]

# ===================== 每个模型的搜索空间（系统化、可复现） =====================
# 注意：键名要与 main_script.py 的 argparse flag 对齐；value 给定候选列表即可。
TUNING_GRID = {
    # ---- Single-Transformer baseline (ST) ----
    "SingleTransformer": {
        "--transformer_layers": [2, 3, 4],
        "--attention_heads":    [2, 4, 8],
        "--ff_dim":             [128, 256],
        "--loss_weight_error":  [0.3, 0.6],
        "--learning_rate":      [1e-4, 2e-4],
        "--dropout_rate":       [0.10, 0.20],
    },

    # ---- DNN (MLP) ----
    # 如果 main_script 接收 --dense_units 作为逗号分隔的单层宽度，你可以传 "128" 或 "256"
    # 若支持多层（如 "128,64"），可在列表里直接放字符串 "128,64"
    "DNN": {
        "--dense_units":   ["64", "128", "256"],   # 或 ["128,64", "256,128"] 如果支持多层
        "--dropout_rate":  [0.0, 0.1, 0.2],
        "--learning_rate": [5e-5, 1e-4, 5e-4],
        "--weight_decay":  [0.0, 1e-5],
        "--activation":    ["relu"],               # 固定为 relu；若要探索可给 ["relu","gelu"]
        "--loss_weight_error": [0.6],              # 与你主体设定一致；如需搜索可开放 [0.3,0.6]
    },

    # ---- LS-SVM （分类）----
    # 传统模型通常没有 epochs；这里只做核参数网格
    # 若 main_script 里的模型名是 "SVM" 或 "LS_SVM"，把此处键改掉，并在 MODEL_NAME_MAP 映射
    "LS-SVM": {
        "--svm_kernel": ["rbf"],  # 固定RBF
        "--svm_C":      [0.1, 0.3, 1, 3, 10],
        "--svm_gamma":  [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        "--svm_class_weight": ["balanced"],
        "--standardize": ["true"],   # 统一做标准化
    },

    # # ---- XGBoost （分类/回归分支若合并在 main_script，中性命名即可）----
    "XGBoost": {
        "--xgb_n_estimators":   [100, 300, 600],
        "--xgb_max_depth":      [3, 5, 7],
        "--xgb_learning_rate":  [0.05, 0.1, 0.2],
        "--xgb_subsample":      [0.7, 1.0],
        "--xgb_colsample_bytree":[0.7, 1.0],
        "--xgb_gamma":          [0.0, 1.0],
        "--xgb_min_child_weight":[1, 5],
    },

    # ---- CNN-LSTM ----
    "CNNLSTM": {
        "--conv_filters":  [32, 64],
        "--kernel_size":   [3, 5],
        "--pool_size":     [2],
        "--lstm_units":    [32, 64],
        "--dense_units":   ["64", "128"],
        "--dropout_rate":  [0.0, 0.2],
        "--learning_rate": [1e-4, 5e-4],
        "--activation":    ["relu"],
        "--loss_weight_error": [0.5, 0.6],
    },
}

# 若 --run_only 期望的内部标识与你这里的键不同，可在此映射
MODEL_NAME_MAP = {
    "SingleTransformer": "SingleTransformer",
    "DNN":               "DNN",
    "LS-SVM":            "SVM",           # 例如 main_script 里用 "SVM"
    "XGBoost":           "XGBoost",
    "CNNLSTM":          "CNNLSTM",
    # "DualTransformer": "DualTransformer",  # 需要的话也可加入
}

# ===================== 调度与记录 =====================
def run_single_tuning_config(model_key, params_tuple, run_id):
    """运行单次超参组合"""
    model_for_cli = MODEL_NAME_MAP.get(model_key, model_key)
    save_dir = BASE_SAVE_DIR / model_key / f"run_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    param_str = " | ".join([f"{k}={v}" for k, v in zip(params_tuple[::2], params_tuple[1::2])])
    print(f"\n--- 🚀 Tuning {model_key} (Run {run_id}) | {param_str} ---")

    # 组装命令
    command = [
        "python", MAIN_SCRIPT,
        "--save_dir", str(save_dir),
        "--run_only", str(model_for_cli),
    ]

    # 深度模型加公共训练/早停参数；传统模型略过（不会报错也行）
    if model_key in {"SingleTransformer", "DNN", "CNNLSTM"}:
        command += list(map(str, COMMON_DEEP_ARGS))

    # 拼接本次超参
    command += [str(p) for p in params_tuple]

    try:
        subprocess.run(command, check=True, text=True, cwd=PROJECT_DIR)
        print(f"   ✅ SUCCESS: Run {run_id} for {model_key} completed.")
        return save_dir
    except subprocess.CalledProcessError as e:
        print(f"   ❌ FAILED: Run {run_id} for {model_key} failed.")
        print(e.stdout)
        print(e.stderr)
        return None

def _pick_score_column(df):
    for c in SCORE_COLUMN_CANDIDATES:
        if c in df.columns:
            return c
    # 没有匹配列时抛错，提醒你检查 pipeline
    raise KeyError(f"No score column found among: {SCORE_COLUMN_CANDIDATES}. Available: {list(df.columns)}")

def find_best_params(all_results_list):
    """分析所有调优结果，找到每个模型的最佳参数，并保存完整结果到CSV。"""
    print("\n" + "*"*80)
    print("🏆 Finding best hyperparameters and saving all results...")

    if not all_results_list:
        print("   [Warning] No results to analyze. Exiting analysis.")
        return

    full_results_df = pd.DataFrame(all_results_list)
    best_params_all_models = {}

    for model_key in TUNING_GRID.keys():
        model_df = full_results_df[full_results_df['Model'] == model_key].copy()
        # 将 Score 转为数值
        model_df['Score'] = pd.to_numeric(model_df['Score'], errors='coerce').fillna(-1)

        if not model_df.empty and model_df['Score'].max() > -1:
            best_run = model_df.loc[model_df['Score'].idxmax()]
            best_score = float(best_run['Score'])
            # 取该模型grid中的列名
            param_cols = list(TUNING_GRID[model_key].keys())
            best_params = {k: best_run.get(k, None) for k in param_cols}

            print(f"\n   🎉 Best parameters for {model_key}:")
            print(f"      Score (validation BA or cv mean): {best_score:.4f}")
            print(f"      Parameters: {best_params}")
            best_params_all_models[model_key] = best_params
        else:
            print(f"\n   Could not determine best parameters for {model_key}.")

    print("\n" + "*"*80)
    print("Hyperparameter tuning summary:")
    for model, params in best_params_all_models.items():
        print(f"   Model: {model}, Best Params: {params}")

    output_path = BASE_SAVE_DIR / "hyperparameter_tuning_full_results.csv"
    full_results_df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"\n✅ Full tuning results saved to: {output_path}")

def read_score_from_summary(summary_file, model_key):
    """从单次运行的 summary.csv 里读取得分（尽量通用）"""
    try:
        df = pd.read_csv(summary_file)
    except Exception as e:
        # 兼容 index_col="Model" 的旧格式
        try:
            df = pd.read_csv(summary_file, index_col="Model")
            df = df.reset_index()
        except Exception:
            raise e

    score_col = _pick_score_column(df)

    # 若 summary 里有 Model 列，则按模型名筛；否则取第一行
    if "Model" in df.columns:
        # 有的 summary 会记录内部模型名（如 "SVM" 而不是 "LS-SVM"），做一次映射兼容
        internal_name = MODEL_NAME_MAP.get(model_key, model_key)
        sub = df[df["Model"].astype(str).str.lower() == str(internal_name).lower()]
        if not sub.empty:
            return float(sub.iloc[0][score_col])
        # 兜底：取全表最大
        return float(df[score_col].max())
    else:
        return float(df.iloc[0][score_col])

# ===================== 主流程 =====================
if __name__ == "__main__":
    script_start_time = time.time()

    BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(f"--- Hyperparameter Tuning Log (Baselines) | Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")

    all_run_results = []

    for model_key, grid in TUNING_GRID.items():
        param_names = list(grid.keys())
        param_values = list(grid.values())
        param_combinations = list(itertools.product(*param_values))

        print(f"\nFound {len(param_combinations)} combinations for {model_key}.")
        run_id_counter = 1

        for combo in param_combinations:
            params_tuple = tuple(itertools.chain(*zip(param_names, combo)))

            run_start_time = time.time()
            save_dir = run_single_tuning_config(model_key, params_tuple, run_id_counter)
            duration_seconds = time.time() - run_start_time
            print(f"   ⏰ Time for this run: {duration_seconds:.2f} seconds ({duration_seconds/60:.2f} minutes).")

            status = "FAILED"
            score = "N/A"
            stopped_epoch = "N/A"
            current_params = {name: val for name, val in zip(param_names, combo)}

            if save_dir:
                # 读取 summary
                try:
                    # 优先读 config 里定义的名称
                    try:
                        from config import SUMMARY_CSV_FILENAME as CFG_SUMMARY
                    except Exception:
                        CFG_SUMMARY = DEFAULT_SUMMARY_CSV
                    summary_file = save_dir / CFG_SUMMARY
                    if summary_file.exists():
                        try:
                            score_val = read_score_from_summary(summary_file, model_key)
                            score = f"{score_val:.6f}"
                            status = "SUCCESS"
                            # Optional: 若 summary 有早停轮数列，可一并读取
                            try:
                                df_tmp = pd.read_csv(summary_file)
                                for candidate in ["stopped_epoch", "stopped_epoch_cv_mean", "early_stop_epoch"]:
                                    if candidate in df_tmp.columns:
                                        stopped_epoch = df_tmp[candidate].iloc[0]
                                        break
                            except Exception:
                                pass
                        except Exception as e:
                            status = f"ERROR_READING_CSV: {e}"
                    else:
                        status = "NO_SUMMARY_FILE"
                except Exception as e:
                    status = f"EXC: {e}"

            # 记录单次
            result_data = {
                "Model": model_key,
                "RunID": run_id_counter,
                "Status": status,
                "Score": score,
                "StoppedEpoch": stopped_epoch,
                **current_params,
                "Duration_sec": round(duration_seconds, 2),
            }
            all_run_results.append(result_data)

            # 规范化早停显示
            stopped_epoch_str = f"{stopped_epoch:.1f}" if isinstance(stopped_epoch, (int, float)) else str(stopped_epoch)
            log_entry = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Model={model_key}, Run={run_id_counter}, Status={status}, "
                f"Duration={duration_seconds:.2f}s, Score={score}, StoppedEpoch={stopped_epoch_str}, "
                f"Params={current_params}\n"
            )
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(log_entry)

            run_id_counter += 1

    # 汇总与最佳
    find_best_params(all_run_results)

    total_duration_minutes = (time.time() - script_start_time) / 60
    print(f"\n✨ Hyperparameter tuning finished. Total time: {total_duration_minutes:.2f} minutes.")
