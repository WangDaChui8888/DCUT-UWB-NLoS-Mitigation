# -*- coding: utf-8 -*-
"""
Model performance evaluation utilities.
MODIFIED to return dictionaries of metrics.
(No changes needed for K-Fold CV support, as functions evaluate given true/pred pairs)
"""
import numpy as np
import pandas as pd
import traceback
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, balanced_accuracy_score,
                             mean_squared_error, mean_absolute_error)
import warnings # Import warnings

def evaluate_classification(y_test_nlos, y_pred_nlos, y_pred_prob_nlos=None): # Added y_pred_prob_nlos for potential future use (e.g., AUC)
    """
    Calculates and prints classification performance metrics.
    Returns a dictionary containing the calculated metrics.
    """
    print("\n--- NLOS 分类性能评估 ---")
    # 初始化用于存储指标的字典
    metrics = {}

    # 检查输入是否有效，如果无效则打印消息并返回空字典
    if y_test_nlos is None or y_pred_nlos is None or len(y_test_nlos) == 0 or len(y_pred_nlos) == 0:
        print("   无法进行分类评估：输入数据不足或无效。")
        return metrics # 返回空字典
    if len(y_test_nlos) != len(y_pred_nlos):
        print(f"   无法进行分类评估：输入数组长度不匹配 (y_true: {len(y_test_nlos)}, y_pred: {len(y_pred_nlos)})。")
        return metrics

    try:
        # 计算基本指标
        accuracy = accuracy_score(y_test_nlos, y_pred_nlos)
        # Balanced Accuracy can fail if only one class is present in y_true
        try:
            balanced_acc = balanced_accuracy_score(y_test_nlos, y_pred_nlos)
        except ValueError:
             print("   [警告] 计算平衡准确率失败（可能真实标签只有一类）。")
             balanced_acc = np.nan # Assign NaN if calculation fails

        # 存储指标
        metrics['Accuracy'] = accuracy
        metrics['Balanced_Accuracy'] = balanced_acc

        # 打印指标
        print(f"   准确率 (Accuracy): {accuracy:.4f}")
        print(f"   平衡准确率 (Balanced Accuracy): {balanced_acc:.4f}")

        # 获取标签信息用于后续计算和报告
        # Ensure labels are derived robustly
        labels_true = np.unique(y_test_nlos)
        labels_pred = np.unique(y_pred_nlos)
        labels = sorted(np.unique(np.concatenate((labels_true, labels_pred))))

        # Determine target names based on standard LOS/NLOS or generic class names
        if all(l in [0, 1] for l in labels):
            target_names = ['LOS (0)', 'NLOS (1)']
            # Ensure target_names correspond correctly to labels if only one label exists
            if len(labels) == 1:
                 target_names = [target_names[labels[0]]]
        else:
             target_names = [f'Class {l}' for l in labels]

        # 计算 NLOS (类别 1) 的 Precision, Recall, F1 (如果存在)
        if 1 in labels_true or 1 in labels_pred: # Check if class 1 is present in either true or pred
            # Use labels=labels to ensure consistency if one class is missing
            # zero_division=0 returns 0.0, use 'warn' to see warnings or 1 to return 1.0
            precision_nlos = precision_score(y_test_nlos, y_pred_nlos, labels=labels, pos_label=1, zero_division=0)
            recall_nlos = recall_score(y_test_nlos, y_pred_nlos, labels=labels, pos_label=1, zero_division=0)
            f1_nlos = f1_score(y_test_nlos, y_pred_nlos, labels=labels, pos_label=1, zero_division=0)
            # 存储指标
            metrics['Precision_NLOS'] = precision_nlos
            metrics['Recall_NLOS'] = recall_nlos
            metrics['F1_NLOS'] = f1_nlos
            # 打印指标
            print(f"   精确率 (Precision, NLOS=1): {precision_nlos:.4f}")
            print(f"   召回率 (Recall, NLOS=1): {recall_nlos:.4f}")
            print(f"   F1分数 (F1-Score, NLOS=1): {f1_nlos:.4f}")
        else:
            # 如果测试集中没有 NLOS 样本 (或预测全为 LOS), 设定默认值
            metrics['Precision_NLOS'] = 0.0
            metrics['Recall_NLOS'] = 0.0
            metrics['F1_NLOS'] = 0.0
            print("   未找到 NLOS (标签 1) 样本或预测，相关指标记为 0。")


        # 计算并打印混淆矩阵
        print("\n   混淆矩阵 (Confusion Matrix):")
        conf_matrix = confusion_matrix(y_test_nlos, y_pred_nlos, labels=labels)
        # Ensure it's 2x2 for TN/FP/FN/TP interpretation
        if conf_matrix.shape == (2, 2) and list(labels) == [0, 1]:
             # 存储 TN, FP, FN, TP
             tn, fp, fn, tp = conf_matrix.ravel()
             metrics['TN'] = tn
             metrics['FP'] = fp
             metrics['FN'] = fn
             metrics['TP'] = tp
             print("       预测 LOS  预测 NLOS")
             print(f"实际 LOS [[{tn:<7} {fp:<7}]] (TN, FP)")
             print(f"实际 NLOS [[{fn:<7} {tp:<7}]] (FN, TP)")
        else:
             # 对于非二分类或特殊情况，只打印矩阵
             print(conf_matrix)
             print(f"(标签顺序: {labels})") # 明确标签顺序

        # 打印分类报告
        print("\n   分类报告 (Classification Report):")
        try:
            # Suppress UndefinedMetricWarning if precision/recall/F1 are ill-defined
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore")
                 # 只打印报告，不存储（因为解析字符串复杂，且主要指标已单独存储）
                 class_report_str = classification_report(y_test_nlos, y_pred_nlos, labels=labels, target_names=target_names, zero_division=0)
                 print(class_report_str)
        except Exception as report_e:
             print(f"   生成分类报告时出错: {report_e}")

        # 在 try 块成功结束时返回收集到的指标字典
        return metrics

    except Exception as e:
        print(f"[错误] 计算分类指标时出错: {e}")
        traceback.print_exc()
        # 发生异常时也返回当前已收集到的指标（可能为空）或只返回空字典
        return metrics # 或者 return {} 保证总返回字典

def evaluate_regression(y_test_error, y_pred_error):
    """
    Calculates and prints regression performance metrics (expects input in mm, reports in m and mm).
    Returns a dictionary containing the calculated metrics.
    """
    print("\n--- 测距误差回归性能评估 ---")
    # 初始化用于存储指标的字典
    metrics = {}

    # 检查输入是否有效，如果无效则打印消息并返回空字典
    if y_test_error is None or y_pred_error is None or len(y_test_error) == 0 or len(y_pred_error) == 0:
        print("   无法进行回归评估：输入数据不足或无效。")
        return metrics # 返回空字典
    if len(y_test_error) != len(y_pred_error):
        print(f"   无法进行回归评估：输入数组长度不匹配 (y_true: {len(y_test_error)}, y_pred: {len(y_pred_error)})。")
        return metrics

    try:
        # --- 计算原始单位(假设是mm)的指标 ---
        # Ensure inputs are numpy arrays for calculations
        y_test_error_mm = np.asarray(y_test_error)
        y_pred_error_mm = np.asarray(y_pred_error)

        # Filter out NaN values before calculation to avoid warnings/errors
        valid_indices = ~np.isnan(y_test_error_mm) & ~np.isnan(y_pred_error_mm)
        if not np.any(valid_indices):
            print("   无法进行回归评估：过滤 NaN 后无有效数据。")
            return metrics
        y_test_valid_mm = y_test_error_mm[valid_indices]
        y_pred_valid_mm = y_pred_error_mm[valid_indices]

        mse_mm2 = mean_squared_error(y_test_valid_mm, y_pred_valid_mm)
        rmse_mm = np.sqrt(mse_mm2)
        mae_mm = mean_absolute_error(y_test_valid_mm, y_pred_valid_mm)

        # --- 将指标转换为米(m)和平方米(m2) ---
        mse_m2 = mse_mm2 / 1_000_000.0 # 使用下划线提高可读性
        rmse_m = rmse_mm / 1000.0
        mae_m = mae_mm / 1000.0

        # --- 存储指标 ---
        metrics['MSE_m2'] = mse_m2
        metrics['RMSE_m'] = rmse_m
        metrics['MAE_m'] = mae_m
        metrics['RMSE_mm'] = rmse_mm # 也存储毫米单位作为参考
        metrics['MAE_mm'] = mae_mm  # 也存储毫米单位作为参考

        # --- 打印以米为单位的结果 ---
        print(f"   均方误差 (MSE, m2): {mse_m2:.6f}")
        print(f"   均方根误差 (RMSE, m): {rmse_m:.4f}")
        print(f"   平均绝对误差 (MAE, m): {mae_m:.4f}")
        print(f"   (原始 RMSE: {rmse_mm:.2f} mm, MAE: {mae_mm:.2f} mm)") # 打印参考值

        # --- 打印样本对比 (使用有效数据) ---
        print("\n   真实误差 vs 预测误差 (前 10 个有效测试样本):")
        num_samples_to_show = min(10, len(y_test_valid_mm))
        if num_samples_to_show > 0:
            print("       样本   真实误差(m)   预测误差(m)     差异(m)")
            print("       ----   -----------   -----------   ----------")
            for i in range(num_samples_to_show):
                y_test_m = y_test_valid_mm[i] / 1000.0
                y_pred_m = y_pred_valid_mm[i] / 1000.0
                diff_m = y_test_m - y_pred_m
                print(f"       {i+1:<4} {y_test_m:>11.4f} {y_pred_m:>11.4f} {diff_m:>11.4f}")
        else:
            print("       测试集在过滤 NaN 后为空。")

        # 在 try 块成功结束时返回收集到的指标字典
        return metrics

    except Exception as e:
        print(f"[错误] 计算回归指标时出错: {e}")
        traceback.print_exc()
        # 发生异常时也返回当前已收集到的指标（可能为空）或只返回空字典
        return metrics # 或者 return {} 保证总返回字典
