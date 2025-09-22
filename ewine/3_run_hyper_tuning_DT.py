import subprocess
from pathlib import Path
import time
import pandas as pd
import itertools
from datetime import datetime

# --- 配置 ---
# TUNING_GRID 是驱动整个脚本的核心
TUNING_GRID = {
    "DualTransformer": {
        # --- 固定已找到的最优值 ---
        '--transformer_layers': [6],
        '--loss_weight_error': [0.6],
        '--learning_rate': [0.0002],
        '--dropout_rate': [0.1],
        
        # --- 【新增】探索前馈网络维度和注意力头数 ---
        '--ff_dim': [128, 256, 512],
        '--attention_heads': [2, 4, 8],
    }
}

MAIN_SCRIPT = "main_script.py"
PROJECT_DIR = Path(__file__).resolve().parent
BASE_SAVE_DIR = PROJECT_DIR / "Tuning_Runs_Round3"
LOG_FILE_PATH = PROJECT_DIR / "tuning_log_round3.txt" 

def run_single_tuning_config(model_name, params_tuple, run_id):
    """运行单次超参数组合"""
    save_dir = BASE_SAVE_DIR / model_name / f"run_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    param_str = " | ".join([f"{k}={v}" for k, v in zip(params_tuple[::2], params_tuple[1::2])])
    print(f"\n--- 🚀 Tuning {model_name} (Run {run_id}) | {param_str} ---")
    
    command = [
        "python", MAIN_SCRIPT,
        "--save_dir", str(save_dir),
        "--run_only", model_name
    ] + [str(p) for p in params_tuple]
    
    try:
        subprocess.run(
            command, check=True, text=True, cwd=PROJECT_DIR
        )
        print(f"   ✅ SUCCESS: Run {run_id} for {model_name} completed.")
        return save_dir
    except subprocess.CalledProcessError as e:
        print(f"   ❌ FAILED: Run {run_id} for {model_name} failed.")
        print(e.stdout)
        print(e.stderr)
        return None

def find_best_params(all_results_list):
    """分析所有调优结果，找到最佳参数，并保存完整结果到CSV。"""
    print("\n" + "*"*80)
    print("🏆 Finding best hyperparameters and saving all results...")
    
    if not all_results_list:
        print("   [Warning] No results to analyze. Exiting analysis.")
        return

    full_results_df = pd.DataFrame(all_results_list)
    best_params_all_models = {}
    
    for model_name in TUNING_GRID.keys():
        model_df = full_results_df[full_results_df['Model'] == model_name].copy()
        model_df['Score'] = pd.to_numeric(model_df['Score'], errors='coerce').fillna(-1)
        
        if not model_df.empty and model_df['Score'].max() > -1:
            best_run = model_df.loc[model_df['Score'].idxmax()]
            best_score = best_run['Score']
            param_cols = [col for col in TUNING_GRID[model_name].keys()]
            best_params = best_run[param_cols].to_dict()
            
            print(f"\n   🎉 Best parameters for {model_name}:")
            print(f"      Score (Balanced Accuracy CV Mean): {best_score:.4f}")
            print(f"      Parameters: {best_params}")
            best_params_all_models[model_name] = best_params
        else:
            print(f"\n   Could not determine best parameters for {model_name}.")

    print("\n" + "*"*80)
    print("Hyperparameter tuning summary:")
    for model, params in best_params_all_models.items():
        print(f"   Model: {model}, Best Params: {params}")
    
    output_path = BASE_SAVE_DIR / "hyperparameter_tuning_full_results.csv"
    full_results_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"\n✅ Full tuning results saved to: {output_path}")

if __name__ == "__main__":
    script_start_time = time.time()
    
    BASE_SAVE_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE_PATH, "w") as f:
        f.write(f"--- Hyperparameter Tuning Log (Round 3) | Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")

    all_run_results = []

    for model_name, grid in TUNING_GRID.items():
        param_names = list(grid.keys())
        param_values = list(grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"\nFound {len(param_combinations)} combinations for {model_name}.")
        
        run_id_counter = 1
        for combo in param_combinations:
            params_tuple = tuple(itertools.chain(*zip(param_names, combo)))
            
            run_start_time = time.time()
            save_dir = run_single_tuning_config(model_name, params_tuple, run_id_counter)
            run_end_time = time.time()
            duration_seconds = run_end_time - run_start_time
            
            print(f"   ⏰ Time for this run: {duration_seconds:.2f} seconds ({duration_seconds/60:.2f} minutes).")

            status = "FAILED"
            score = -1.0
            stopped_epoch = 'N/A' # 【新增】初始化早停轮数变量
            current_params = {name: val for name, val in zip(param_names, combo)}
            
            if save_dir:
                try:
                    from config import SUMMARY_CSV_FILENAME
                except ImportError:
                    SUMMARY_CSV_FILENAME = "summary.csv"
                
                summary_file = save_dir / SUMMARY_CSV_FILENAME
                if summary_file.exists():
                    try:
                        df = pd.read_csv(summary_file, index_col="Model")
                        score = df.loc[model_name, 'Balanced_Accuracy_cv_mean']
                        # 【新增】尝试读取早停轮数，如果列不存在则保持'N/A'
                        if 'stopped_epoch_cv_mean' in df.columns:
                            stopped_epoch = df.loc[model_name, 'stopped_epoch_cv_mean']
                        status = "SUCCESS"
                    except Exception as e:
                        print(f"   [Error] Could not read results from {summary_file}: {e}")
                        status = "ERROR_READING_CSV"
                else:
                    status = "NO_SUMMARY_FILE"
            
            result_data = {
                'Model': model_name,
                'RunID': run_id_counter,
                'Status': status,
                'Score': score if score != -1.0 else 'N/A',
                'StoppedEpoch': stopped_epoch, # 【新增】将早停信息添加到结果字典
                **current_params,
                'Duration_sec': round(duration_seconds, 2)
            }
            all_run_results.append(result_data)

            # 在 run_hyperparameter_tuning.py 中，找到并替换这部分代码

            # 【修改】先处理 StoppedEpoch 的显示格式，再创建日志条目
            stopped_epoch_val = result_data['StoppedEpoch']
            if isinstance(stopped_epoch_val, (int, float)):
                stopped_epoch_str = f"{stopped_epoch_val:.1f}"
            else:
                stopped_epoch_str = str(stopped_epoch_val) # 如果是 'N/A'，则直接转为字符串

            log_entry = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Model={model_name}, Run={run_id_counter}, Status={status}, "
                f"Duration={duration_seconds:.2f}s, Score={result_data['Score']}, "
                f"StoppedEpoch={stopped_epoch_str}, " # 使用处理好的字符串
                f"Params={current_params}\n"
            )
            with open(LOG_FILE_PATH, "a", encoding='utf-8') as f:
                f.write(log_entry)
            
            run_id_counter += 1

    find_best_params(all_run_results)
    
    script_end_time = time.time()
    total_duration_minutes = (script_end_time - script_start_time) / 60
    print(f"\n✨ Hyperparameter tuning finished. Total time: {total_duration_minutes:.2f} minutes.")