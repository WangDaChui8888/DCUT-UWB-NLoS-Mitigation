# -*- coding: utf-8 -*-
"""
用于从保存的 .mat 文件计算和显示性能指标的脚本。

此脚本读取由主 K 折交叉验证脚本保存的最终测试集结果
（特别是 .mat 文件）并计算关键的回归指标。
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import sys

# --- 配置 ---
# 重要提示：请将此设置为与您的 config.py 中使用的 *完全相同* 的目录
#            或者主脚本保存结果的位置。
# 示例： SAVE_DIRECTORY = Path("./uwb_results")
#        或 SAVE_DIRECTORY = Path("C:/Users/YourUser/Documents/UWP_Project/Results")
try:
    # 尝试导入原始的 config 文件（如果它在 Python 路径中）
    import config
    SAVE_DIRECTORY = config.SAVE_DIRECTORY
    print(f"信息：已从 config.py 加载 SAVE_DIRECTORY: {SAVE_DIRECTORY}") # 翻译: INFO

    # --- 从 config 获取文件名（使用 getattr 并提供默认值） ---
    # 定义默认名称，以防 config 属性丢失
    default_names = {
        "DualTransformer": ("dualtransformer_results_final_test.mat", "DT"),
        "DNN": ("dnn_results_final_test.mat", "DNN"),
        "SingleTransformer": ("singletransformer_results_final_test.mat", "ST"),
        "SVM": ("svc_ridge_results_final_test.mat", "LS-SVM"), # 键是 SVM，显示名称是 LS-SVM
    }

    # 将内部模型键映射到文件名和显示名称
    MODEL_FILES_AND_DISPLAY = {
        "DualTransformer": (
            getattr(config, 'DUAL_TRANSFORMER_FINAL_RESULTS_MAT', default_names["DualTransformer"][0]),
            default_names["DualTransformer"][1]
        ),
        "DNN": (
            getattr(config, 'DNN_FINAL_RESULTS_MAT', default_names["DNN"][0]),
            default_names["DNN"][1]
        ),
        "SingleTransformer": (
            getattr(config, 'SINGLE_TRANSFORMER_FINAL_RESULTS_MAT', default_names["SingleTransformer"][0]),
            default_names["SingleTransformer"][1]
        ),
        "SVM": ( # main.py 中用于 SVC+Ridge 的内部键 "SVM"
            getattr(config, 'SVC_RIDGE_FINAL_RESULTS_MAT', default_names["SVM"][0]),
             default_names["SVM"][1]
        ),
    }
    print("信息：已从 config.py 加载结果文件名") # 翻译: INFO

except ImportError:
    print("警告：无法导入 config.py。将使用默认保存目录 './results' 和默认文件名。") # 翻译: WARNING
    print("         如果需要，请手动修改 SAVE_DIRECTORY 和 MODEL_FILES_AND_DISPLAY。")
    SAVE_DIRECTORY = Path("./results") # 备用目录
    MODEL_FILES_AND_DISPLAY = {
        # 内部键: (文件名, 显示名称)
        "DualTransformer": ("dualtransformer_results_final_test.mat", "DT"),
        "DNN": ("dnn_results_final_test.mat", "DNN"),
        "SingleTransformer": ("singletransformer_results_final_test.mat", "ST"),
        "SVM": ("svc_ridge_results_final_test.mat", "LS-SVM"), # 代表 SVC+Ridge
    }
except AttributeError as e:
     print(f"警告：在 config.py 中找不到 SAVE_DIRECTORY ({e})。将使用默认值 './results'。") # 翻译: WARNING
     print("          如果需要，请手动修改 SAVE_DIRECTORY。")
     SAVE_DIRECTORY = Path("./results") # 备用目录
     # 如果只是 SAVE_DIRECTORY 属性丢失，则尝试再次加载文件名（使用默认值）
     try:
         default_names = {
            "DualTransformer": ("dualtransformer_results_final_test.mat", "DT"),
            "DNN": ("dnn_results_final_test.mat", "DNN"),
            "SingleTransformer": ("singletransformer_results_final_test.mat", "ST"),
            "SVM": ("svc_ridge_results_final_test.mat", "LS-SVM"),
         }
         MODEL_FILES_AND_DISPLAY = {
            "DualTransformer": (getattr(config, 'DUAL_TRANSFORMER_FINAL_RESULTS_MAT', default_names["DualTransformer"][0]), default_names["DualTransformer"][1]),
            "DNN": (getattr(config, 'DNN_FINAL_RESULTS_MAT', default_names["DNN"][0]), default_names["DNN"][1]),
            "SingleTransformer": (getattr(config, 'SINGLE_TRANSFORMER_FINAL_RESULTS_MAT', default_names["SingleTransformer"][0]), default_names["SingleTransformer"][1]),
            "SVM": (getattr(config, 'SVC_RIDGE_FINAL_RESULTS_MAT', default_names["SVM"][0]), default_names["SVM"][1]),
         }
         print("信息：已从 config.py 加载结果文件名（在需要时使用默认值）。") # 翻译: INFO
     except NameError: # config 本身未导入
         print("警告：找不到 config.py。将使用默认文件名。") # 翻译: WARNING
         MODEL_FILES_AND_DISPLAY = {
            "DualTransformer": ("dualtransformer_results_final_test.mat", "DT"),
            "DNN": ("dnn_results_final_test.mat", "DNN"),
            "SingleTransformer": ("singletransformer_results_final_test.mat", "ST"),
            "SVM": ("svc_ridge_results_final_test.mat", "LS-SVM"),
         }


# main.py 在保存 .mat 文件时用于替换 NaN 的占位符值
NAN_PLACEHOLDER = -999.0

# --- 用于指标计算的辅助函数 ---
def calculate_regression_metrics(y_true_m, y_pred_m):
    """
    计算回归指标（MAE、RMSE、STD、最大绝对误差、最小绝对误差），单位为毫米 (mm)。

    Args:
        y_true_m (np.ndarray): 真实误差值数组（单位：米）。
        y_pred_m (np.ndarray): 预测误差值数组（单位：米）。

    Returns:
        dict: 包含计算出的指标（单位 mm）的字典，如果输入无效则返回 None。
              键: 'MAE (mm)', 'RMSE (mm)', 'STD (mm)',
                  'Max Error (mm)', 'Min Error (mm)'
    """
    if y_true_m is None or y_pred_m is None or len(y_true_m) != len(y_pred_m) or len(y_true_m) == 0:
        print("错误：用于指标计算的输入数组无效。") # 翻译: ERROR
        return None

    # 确保是 numpy 数组并展平
    y_true_m = np.asarray(y_true_m).flatten()
    y_pred_m = np.asarray(y_pred_m).flatten()

    # --- 过滤掉无效条目（NaN 占位符） ---
    # 为有效的预测条目创建布尔掩码
    valid_mask = (y_pred_m != NAN_PLACEHOLDER) & (~np.isnan(y_pred_m)) & (~np.isnan(y_true_m))

    if not np.any(valid_mask):
        print("错误：未找到有效的（非占位符、非 NaN）预测条目。") # 翻译: ERROR
        return {
            'MAE (mm)': np.nan, 'RMSE (mm)': np.nan, 'STD (mm)': np.nan,
            'Max Error (mm)': np.nan, 'Min Error (mm)': np.nan
        }

    # 应用掩码
    y_true_m_valid = y_true_m[valid_mask]
    y_pred_m_valid = y_pred_m[valid_mask]
    print(f"调试：原始长度：{len(y_true_m)}，过滤后有效数据对：{len(y_true_m_valid)}") # 翻译: DEBUG


    # 转换为毫米
    y_true_mm = y_true_m_valid * 1000.0
    y_pred_mm = y_pred_m_valid * 1000.0

    # 计算预测误差 (真实值 - 预测值)，单位 mm
    prediction_errors_mm = y_true_mm - y_pred_mm

    # 计算绝对预测误差，单位 mm
    abs_prediction_errors_mm = np.abs(prediction_errors_mm)

    # 计算指标
    mae_mm = np.mean(abs_prediction_errors_mm)
    rmse_mm = np.sqrt(np.mean(np.square(prediction_errors_mm)))
    std_mm = np.std(prediction_errors_mm) # 预测误差的标准差
    max_abs_error_mm = np.max(abs_prediction_errors_mm)
    min_abs_error_mm = np.min(abs_prediction_errors_mm) # 绝对误差的最小值

    return {
        'MAE (mm)': mae_mm,
        'RMSE (mm)': rmse_mm,
        'STD (mm)': std_mm,
        'Max Error (mm)': max_abs_error_mm,
        'Min Error (mm)': min_abs_error_mm, # 您要求的 Min Error（最小绝对误差）
    }

# --- 主要处理流程 ---
results_summary = {} # 用于存储所有模型结果的字典

print(f"\n--- 正在处理来自以下目录的结果： {SAVE_DIRECTORY} ---")

# 遍历映射中定义的模型
for model_key, (mat_filename, display_name) in MODEL_FILES_AND_DISPLAY.items():
    print(f"\n正在处理模型： {display_name} ({model_key})") # 翻译: Processing Model
    mat_filepath = SAVE_DIRECTORY / mat_filename # 构建 .mat 文件的完整路径

    # 检查文件是否存在
    if not mat_filepath.exists():
        print(f"  警告：结果文件未找到： {mat_filepath}") # 翻译: WARNING
        results_summary[display_name] = { # 使用显示名称作为表格索引
            'MAE (mm)': np.nan, 'RMSE (mm)': np.nan, 'STD (mm)': np.nan,
            'Max Error (mm)': np.nan, 'Min Error (mm)': np.nan
        }
        continue # 跳过当前模型，处理下一个

    try:
        # 从 .mat 文件加载数据
        mat_data = loadmat(str(mat_filepath))
        print(f"  成功加载： {mat_filepath.name}") # 翻译: Successfully loaded

        # 提取所需的数组（这里的误差单位预期是米）
        y_test_error_m = mat_data.get('y_test_error', None) # 真实误差（米）
        y_pred_error_m = mat_data.get('y_pred_error', None) # 预测误差（米）

        if y_test_error_m is None or y_pred_error_m is None:
            print(f"  错误：在 {mat_filename} 中未找到键 'y_test_error' 或 'y_pred_error'") # 翻译: ERROR
            metrics = None
        else:
             # 确保数组是一维的
            y_test_error_m = y_test_error_m.flatten()
            y_pred_error_m = y_pred_error_m.flatten()
            print(f"  已提取 'y_test_error' (形状: {y_test_error_m.shape}) 和 'y_pred_error' (形状: {y_pred_error_m.shape})") # 翻译: Extracted...

            # 计算指标
            metrics = calculate_regression_metrics(y_test_error_m, y_pred_error_m)

        if metrics:
            print(f"  计算出的指标 (mm): {metrics}") # 翻译: Calculated Metrics
            results_summary[display_name] = metrics # 使用显示名称作为索引
        else:
             print(f"  未能计算模型 {display_name} 的指标。") # 翻译: Failed to calculate metrics for...
             results_summary[display_name] = { # 使用显示名称作为索引
                 'MAE (mm)': np.nan, 'RMSE (mm)': np.nan, 'STD (mm)': np.nan,
                 'Max Error (mm)': np.nan, 'Min Error (mm)': np.nan
             }


    except Exception as e:
        print(f"  错误：处理文件 {mat_filepath.name} 失败： {e}") # 翻译: ERROR: Failed to process file...
        results_summary[display_name] = { # 使用显示名称作为索引
             'MAE (mm)': np.nan, 'RMSE (mm)': np.nan, 'STD (mm)': np.nan,
             'Max Error (mm)': np.nan, 'Min Error (mm)': np.nan
        }

# --- 显示结果 ---
if not results_summary:
    print("\n没有处理任何结果。") # 翻译: No results were processed.
else:
    # 将摘要字典转换为 DataFrame
    results_df = pd.DataFrame.from_dict(results_summary, orient='index')
    results_df.index.name = 'Model' # 设置索引列的名称

    # 确保列的顺序正确
    desired_columns = ['MAE (mm)', 'RMSE (mm)', 'STD (mm)', 'Max Error (mm)', 'Min Error (mm)']
    # 重新索引，如果某个指标对所有模型都未计算，则添加值为 NaN 的列
    results_df = results_df.reindex(columns=desired_columns)

    print("\n--- 最终指标摘要 ---") # 翻译: Final Metrics Summary
    # 格式化输出以便更好地阅读
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 1000,
                           'display.float_format', '{:.1f}'.format): # 格式化为一位小数，与图片类似
        print(results_df)

    # --- 可选：将摘要表格保存到 CSV 文件 ---
    summary_save_path = SAVE_DIRECTORY / "calculated_metrics_summary.csv" # 定义保存路径
    try:
        results_df.to_csv(summary_save_path, float_format='%.4f') # 保存时使用更高精度
        print(f"\n摘要表格已保存至： {summary_save_path}") # 翻译: Summary table saved to:
    except Exception as e:
        print(f"\n错误：保存摘要 CSV 文件失败： {e}") # 翻译: ERROR: Failed to save summary CSV:

print("\n--- 脚本执行完毕 ---") # 翻译: Script Finished