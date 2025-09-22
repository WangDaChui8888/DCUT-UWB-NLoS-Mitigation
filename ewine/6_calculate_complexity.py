# -*- coding: utf-8 -*-
"""
calculate_complexity.py (v3.4 - 修复 NameError)

改进点：
1) 修复了 logging 配置中因缺少一行代码导致的 NameError。
2) 内存测量统一改为更稳定的“静态模型文件大小”。
3) 修复了因部分模型分析失败导致结果错位的问题。
"""

import sys
from pathlib import Path
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from memory_profiler import memory_usage
import logging
from collections import defaultdict
import pickle

# --- 将脚本自身所在的目录添加到 Python 搜索路径 ---
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# --- 依赖项目内模块 ---
try:
    from model_definition import DualChannelTransformerModel, PositionalEmbedding
except ImportError as e:
    print(f"❌ 错误: 无法从 'model_definition.py' 导入自定义层或模型。 ({e})")
    sys.exit(1)

try:
    import config
except ImportError:
    print("❌ 错误: 找不到 config.py。")
    sys.exit(1)

# --- 日志配置 ---
def setup_logging():
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-7s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    log_file = SCRIPT_DIR / "complexity_analysis_statistical.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # =========================================================
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 【重要】恢复被删除的代码 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 恢复结束 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # =========================================================
    
    logging.info(f"日志系统已初始化，日志将记录到: {log_file}")

# --- 结果目录 ---
BASE_RESULTS_DIR = SCRIPT_DIR / "Ghent_Statistical_Runs"
RUN_DIRS = []
if BASE_RESULTS_DIR.exists():
    RUN_DIRS = sorted([d for d in BASE_RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith('run_seed_')])

if not RUN_DIRS:
    # 尝试寻找 Ablation_Study_Runs_Advanced 目录作为备选
    BASE_RESULTS_DIR_ALT = SCRIPT_DIR / "Ablation_Study_Runs_Advanced"
    if BASE_RESULTS_DIR_ALT.exists():
        subdirs = [d for d in BASE_RESULTS_DIR_ALT.iterdir() if d.is_dir()]
        if subdirs:
            # 进入第一层子目录寻找 seed 文件夹
            RUN_DIRS = sorted([d for d in subdirs[0].iterdir() if d.is_dir() and d.name.startswith('seed_')])
            BASE_RESULTS_DIR = subdirs[0] # 更新基础目录

if not RUN_DIRS:
    logging.error(f"❌ 错误: 在 '{BASE_RESULTS_DIR}' 或备选目录中没有找到任何 'run_seed_*' 或 'seed_*' 文件夹。")
    sys.exit(1)

# --- 降噪 ---
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


# ========================= 核心工具函数 =========================
def get_flops_for_tf_model(model, concrete_input_shapes):
    try:
        input_specs = [tf.TensorSpec(shape, dtype=tf.float32) for shape in concrete_input_shapes]
        @tf.function
        def model_fn(*inputs):
            x = inputs if len(inputs) > 1 else inputs[0]
            return model(x, training=False)
        concrete_func = model_fn.get_concrete_function(*input_specs)
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        with tf.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = (tf.compat.v1.profiler.ProfileOptionBuilder(tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
                    .select(['float_ops']).order_by('float_ops').build())
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return float(flops.total_float_ops if flops else 0)
    except Exception:
        return 0.0

# ========================= 分析函数 (返回原始数值, 内存改为静态大小) =========================
def analyze_tf_model(run_dir, model_name, model_filename, custom_objects, concrete_input_shapes):
    model_path = run_dir / model_filename
    if not model_path.exists():
        logging.warning(f"  在 {run_dir.name} 中, 模型文件缺失: {model_filename}，跳过。")
        return None
    try:
        logging.info(f"--- 正在分析 TensorFlow 模型: {model_name} ---")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        params_k = model.count_params() / 1e3
        mflops = get_flops_for_tf_model(model, concrete_input_shapes) / 1e6

        mem_kb = model_path.stat().st_size / 1024
        
        logging.info(f"  分析完成: Parameters={params_k:.2f}K, FLOPs={mflops:.2f}M, Model Size={mem_kb:.2f}KB")
        return {
            "Parameters (K)": params_k,
            "FLOPs (M)": mflops,
            "Model Size (KB)": mem_kb
        }
    except Exception as e:
        logging.error(f"  分析 {model_name} ({run_dir.name}) 时出错: {e}", exc_info=False)
        return None

def analyze_sklearn_model(run_dir, model_name):
    logging.info(f"--- 正在分析 Scikit-learn 模型: {model_name} ---")
    try:
        model_paths = {}
        if model_name == "LS-SVM":
            model_paths = {"cls": run_dir / config.SVC_FINAL_MODEL_FILE, "reg": run_dir / config.RIDGE_FINAL_MODEL_FILE}
        elif model_name == "XGBoost":
            model_paths = {"cls": run_dir / config.XGBOOST_CLS_FINAL_MODEL_FILE, "reg": run_dir / config.XGBOOST_REG_FINAL_MODEL_FILE}
        else: return None
            
        params, mem_kb = 0, 0
        for key, path in model_paths.items():
            if not path.exists():
                logging.warning(f"  在 {run_dir.name} 中, {model_name} 的 {path.name} 文件缺失，跳过。")
                return None
            model_obj = joblib.load(path)
            mem_kb += path.stat().st_size / 1024
            
            if model_name == "LS-SVM" and key == "cls":
                params = model_obj.n_support_.sum()
            elif model_name == "XGBoost":
                params += model_obj.get_booster().num_boosted_rounds()
        
        param_metric = "Parameters (SVs)" if model_name == "LS-SVM" else "Parameters (Trees)"
        
        logging.info(f"  分析完成: {param_metric.split(' ')[0]}={params}, Model Size={mem_kb:.2f}KB")
        return {param_metric: params, "Model Size (KB)": mem_kb}
    except Exception as e:
        logging.error(f"  分析模型 {model_name} ({run_dir.name}) 时出错: {e}", exc_info=False)
        return None

# ========================= 主流程 =========================
if __name__ == "__main__":
    setup_logging()
    logging.info("=" * 60)
    logging.info("🚀 模型复杂度统计分析脚本 v3.4 (终极修复版) 启动")
    logging.info(f"   将分析以下 {len(RUN_DIRS)} 个目录: {[d.name for d in RUN_DIRS]}")
    logging.info("=" * 60)

    all_results = defaultdict(lambda: defaultdict(list))
    custom_objects = {"PositionalEmbedding": PositionalEmbedding, "DualChannelTransformerModel": DualChannelTransformerModel}

    for run_dir in RUN_DIRS:
        logging.info(f"--- 正在处理目录: {run_dir.name} ---")
        
        num_cir_features = len(config.INITIAL_CIR_FEATURES)
        shape_single = [(1, num_cir_features)]
        flos_dim = 0
        try:
            gmm_los = joblib.load(run_dir / config.GMM_LOS_FILENAME)
            gmm_nlos = joblib.load(run_dir / config.GMM_NLOS_FILENAME)
            flos_dim = int(gmm_los.n_components) + int(gmm_nlos.n_components)
        except Exception:
            logging.warning(f"  在 {run_dir.name} 中找不到GMM文件，无法分析 DCUT。")
        
        shape_dual = [(1, num_cir_features), (1, flos_dim)] if flos_dim > 0 else []

        model_analysis_functions = {
            "DCUT": lambda: analyze_tf_model(run_dir, "DCUT", config.DUAL_TRANSFORMER_FINAL_MODEL_FILE, custom_objects, shape_dual),
            "SingleTransformer": lambda: analyze_tf_model(run_dir, "SingleTransformer", config.SINGLE_TRANSFORMER_FINAL_MODEL_FILE, custom_objects, shape_single),
            "DNN": lambda: analyze_tf_model(run_dir, "DNN", config.DNN_FINAL_MODEL_FILE, {}, shape_single),
            "CNN-LSTM": lambda: analyze_tf_model(run_dir, "CNN-LSTM", config.CNN_LSTM_FINAL_MODEL_FILE, {}, shape_single),
            "LS-SVM": lambda: analyze_sklearn_model(run_dir, "LS-SVM"),
            "XGBoost": lambda: analyze_sklearn_model(run_dir, "XGBoost"),
        }
        
        for name, func in model_analysis_functions.items():
            if name == "DCUT" and not shape_dual: continue
            res = func()
            if res:
                for metric, value in res.items():
                    all_results[name][metric].append(value)

    if not all_results:
        logging.warning("❌ 未能成功分析任何模型。请检查模型文件是否存在。")
    else:
        final_summary = []
        model_order = ["DCUT", "SingleTransformer", "DNN", "CNN-LSTM", "LS-SVM", "XGBoost"]
        
        for model_name in model_order:
            if model_name not in all_results: continue
            
            data = all_results[model_name]
            row = {"Model": model_name}
            
            for metric, values in sorted(data.items()):
                if not values: continue
                mean = np.mean(values)
                std = np.std(values)
                row[metric] = f"{mean:.2f} ± {std:.2f}" if len(values) > 1 and std > 1e-9 else f"{mean:.2f}"
            final_summary.append(row)

        summary_df = pd.DataFrame(final_summary).set_index("Model")
        
        logging.info("\n" + "=" * 80)
        logging.info("📊 模型复杂度最终汇总 (均值 ± 标准差)")
        logging.info("=" * 80)
        for line in summary_df.to_string().split('\n'): logging.info(line)
        logging.info("=" * 80)

        csv_path = BASE_RESULTS_DIR / "model_complexity_summary_statistical.csv"
        try:
            summary_df.to_csv(csv_path)
            logging.info(f"✅ 汇总表格已成功保存到: {csv_path}")
        except Exception as e:
            logging.error(f"❌ 保存 CSV 文件失败: {e}")

        logging.info("\n说明:")
        logging.info(" - 所有数值均基于对多个随机种子运行结果的统计分析。")
        logging.info(" - Parameters (K/SVs/Trees): K=千(1e3), SVs=支持向量数, Trees=树的数量。")
        logging.info(" - FLOPs (M): M=兆(1e6)。表示单次推理的计算量（batch=1）。")
        logging.info(" - Model Size (KB): 模型在磁盘上的静态存储大小。")

    logging.info("🚀 统计分析脚本执行完毕。")