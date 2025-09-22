# -*- coding: utf-8 -*-
"""
Data loading, cleaning, preprocessing, splitting, and weighting utilities.
Supports K-Fold Cross-Validation by providing the initial train/test split.
K-Fold splitting itself is handled in the main training script on the training set.
"""
import pandas as pd
import numpy as np
import traceback
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
# from scipy.stats import norm # 不需要了

def load_data(file_path):
    """Loads data from a CSV file."""
    print(f"\n[数据加载] 正在加载数据: {file_path}")
    try:
        data = pd.read_csv(file_path)
        print(f"[数据加载] 成功。 数据形状: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"[错误] 文件未找到! 路径: '{file_path}'")
        return None
    except Exception as e:
        print(f"[错误] 加载数据时出错: {e}")
        traceback.print_exc()
        return None

def clean_data(df, required_cols, cir_cols, z_threshold):
    """Performs data cleaning: NaN, duplicates, column check, outliers."""
    print("\n[数据处理] === 开始数据清洗 ===")
    initial_rows = df.shape[0]
    print(f"[数据处理] 初始行数: {initial_rows}")

    if df is None:
        return None

    # 处理 NaN
    rows_before_nan = df.shape[0]
    df = df.dropna()
    rows_after_nan = df.shape[0]
    if rows_before_nan > rows_after_nan:
        print(f"[数据处理] 移除了 {rows_before_nan - rows_after_nan} 行含NaN值数据。")
    else:
        print("[数据处理] 未发现NaN值。")

    # 处理重复行
    rows_before_dup = df.shape[0]
    df = df.drop_duplicates()
    rows_after_dup = df.shape[0]
    if rows_before_dup > rows_after_dup:
        print(f"[数据处理] 移除了 {rows_before_dup - rows_after_dup} 行重复数据。")
    else:
        print("[数据处理] 未发现重复行。")

    print(f"[数据处理] 清洗后形状 (去重后): {df.shape}")

    # 检查必需列
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[错误] 缺少必需列: {missing_cols}")
        return None
    else:
        print("[数据处理] 所有必需列均存在。")

    # 处理 'error' 列异常值
    print(f"[数据处理] 正在处理 'error' 列异常值 (Z-score < {z_threshold})...")
    rows_before_err_outlier = df.shape[0]
    error_mean = df['error'].mean()
    error_std = df['error'].std()
    if error_std > 1e-6: # 避免除以零或非常小的标准差
        error_z_scores = np.abs((df['error'] - error_mean) / error_std)
        df = df[error_z_scores < z_threshold].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        rows_after_err_outlier = df.shape[0]
    else:
        print("[数据处理] 'error' 列标准差过小或为零，跳过Z-score异常值处理。")
        rows_after_err_outlier = rows_before_err_outlier

    if rows_before_err_outlier > rows_after_err_outlier:
        print(f"[数据处理] 基于 'error' Z-score 移除了 {rows_before_err_outlier - rows_after_err_outlier} 行。")
    else:
        print("[数据处理] 未发现 'error' 异常值。")

    df = df.reset_index(drop=True) # 重置索引

    # 处理 CIR 特征列异常值 (整行移除)
    print(f"[数据处理] 正在处理 CIR 特征列异常值 (Z-score < {z_threshold}，整行移除)...")
    rows_before_cir_outlier = df.shape[0]
    features_cir_df_temp = df[cir_cols]
    try:
        if not features_cir_df_temp.empty:
            features_mean = features_cir_df_temp.mean()
            features_std = features_cir_df_temp.std()
            valid_std_cols = features_std[features_std > 1e-6].index # 只对标准差足够大的列计算 Z-score
            if not valid_std_cols.empty:
                z_scores = np.abs((features_cir_df_temp[valid_std_cols] - features_mean[valid_std_cols]) / features_std[valid_std_cols])
                # 保留所有有效特征的 Z-score 都小于阈值的行
                filtered_indices = (z_scores < z_threshold).all(axis=1)
                df = df[filtered_indices].copy() # 使用 .copy()
                rows_after_cir_outlier = df.shape[0]
            else:
                print("[数据处理] 所有 CIR 特征标准差过小或为零，跳过Z-score异常值处理。")
                rows_after_cir_outlier = rows_before_cir_outlier
        else:
            print("[数据处理] CIR 特征 DataFrame 为空，跳过Z-score异常值处理。")
            rows_after_cir_outlier = rows_before_cir_outlier
    except Exception as e:
        print(f"[警告] 处理 CIR 特征异常值时出错: {e}")
        rows_after_cir_outlier = rows_before_cir_outlier # 出错时不改变数据

    if rows_before_cir_outlier > rows_after_cir_outlier:
        print(f"[数据处理] 基于 CIR 特征 Z-score 移除了 {rows_before_cir_outlier - rows_after_cir_outlier} 行。")
    else:
        print("[数据处理] 未发现 CIR 特征异常值行。")

    df = df.reset_index(drop=True) # 再次重置索引

    print(f"[数据处理] 清洗完成。最终数据形状: {df.shape}")
    if df.empty:
        print("[错误] 清洗后无剩余数据！")
        return None
    return df

def map_labels_and_check(target_nlos_series):
    """Maps LOS/NLOS labels to 0/1 and checks for unknown values."""
    print("\n[数据处理] === 开始标签映射与检查 ===")
    target_nlos_series = target_nlos_series.astype(str).str.strip().str.upper()
    nlos_map = {'NLOS': 1, 'LOS': 0}
    unique_values = target_nlos_series.unique()

    if not all(item in nlos_map for item in unique_values):
        print(f"[错误] 'NLOS' 列包含未知值: {unique_values}. 请确保只有 'LOS' 和 'NLOS'。")
        return None

    try:
        target_nlos_numeric = target_nlos_series.map(nlos_map).astype(int)
        print("[数据处理] NLOS 标签映射完成。")
        print(f"[数据处理] NLOS 标签分布 (0=LOS, 1=NLOS):\n{target_nlos_numeric.value_counts(normalize=True)}")
        return target_nlos_numeric
    except Exception as e:
        print(f"[错误] 映射NLOS标签时出错: {e}")
        traceback.print_exc()
        return None

# --- 移除 align_features 函数，因为它在 main_script 中未使用 ---

# --- 修改后的 prepare_numpy_arrays 函数 ---
def prepare_numpy_arrays(features_cir_df, target_nlos_numeric, error_series):
    """
    将 CIR 特征、NLOS 标签和 Error 值从 pandas 对象转换为 NumPy 数组。
    (不再处理 FLOS 特征)
    """
    print("\n[数据准备] === 准备模型输入 NumPy 数组 (CIR, NLOS, Error) ===")
    try: # 添加 try-except 块增加健壮性
        X_cir = features_cir_df.values
        y_nlos = target_nlos_numeric.values
        y_error = error_series.values

        # 更新打印信息，移除 X_flos
        print(f"[调试信息] 数组形状: X_cir={X_cir.shape}, y_nlos={y_nlos.shape}, y_error={y_error.shape}")

        return X_cir, y_nlos, y_error

    except AttributeError as ae:
         print(f"[错误] 转换到 NumPy 数组时出错: 输入可能不是 Pandas 对象 ({ae})")
         traceback.print_exc()
         return None, None, None # 在出错时返回 None
    except Exception as e:
         print(f"[错误] 转换到 NumPy 数组时发生未知错误: {e}")
         traceback.print_exc()
         return None, None, None # 在出错时返回 None
# --- 修改结束 ---


def split_data(X_cir, y_nlos, y_error, test_size, random_state):
    """
    Splits CIR, labels, errors into initial training and final testing sets.
    K-Fold splitting will be performed on the returned training set later.
    """
    print(f"\n[数据划分] === 划分初始训练/最终测试集 (测试集比例: {test_size}, 随机种子: {random_state}) ===")
    print("[数据划分] 注意: 此处划分的测试集将作为最终评估集，不参与 K 折交叉验证。")
    try:
        split_result = train_test_split(
            X_cir, y_nlos, y_error, # 只划分这三个
            test_size=test_size, random_state=random_state, stratify=y_nlos)
        print("[数据划分] 完成 (使用分层抽样)。")
    except ValueError as e:
        print(f"[警告] 分层抽样失败 ({e})，尝试普通随机抽样...")
        split_result = train_test_split(
            X_cir, y_nlos, y_error, test_size=test_size, random_state=random_state)
        print("[数据划分] 完成 (未使用普通随机抽样)。")
    except Exception as e_split: # 捕获其他潜在错误
        print(f"[错误] 数据划分时发生意外错误: {e_split}")
        traceback.print_exc()
        return None # 返回 None 表示失败

    X_train_cir, X_test_cir, y_train_nlos, y_test_nlos, y_train_error, y_test_error = split_result
    print(f"[数据划分] 初始训练集大小: {len(y_train_nlos)}, 最终测试集大小: {len(y_test_nlos)}")
    print(f"[数据划分] 初始训练集 NLOS 标签分布 (0=LOS, 1=NLOS):\n{pd.Series(y_train_nlos).value_counts(normalize=True)}")
    print(f"[数据划分] 最终测试集 NLOS 标签分布 (0=LOS, 1=NLOS):\n{pd.Series(y_test_nlos).value_counts(normalize=True)}")
    # 只返回这 6 个数组
    return X_train_cir, X_test_cir, y_train_nlos, y_test_nlos, y_train_error, y_test_error

def calculate_class_weights(y_train):
    """
    Calculates class weights for imbalanced classification.
    Note: In K-Fold CV, this might be called inside the loop for each training fold.
    """
    print("\n[类别权重] === 计算分类任务的类别权重 ===")
    unique_classes = np.unique(y_train)
    class_weights_dict = None
    if len(unique_classes) > 1:
        try:
            weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weights_dict = dict(zip(unique_classes, weights))
            print(f"[类别权重] 计算得到的权重: {class_weights_dict}")
        except Exception as e:
            print(f"[错误] 计算类别权重时出错: {e}")
            traceback.print_exc() # 打印错误细节
    else:
        print("[警告] 训练数据中只存在一个类别，无法计算类别权重。")
    return class_weights_dict

# --- 删除 compute_feature_stats 和 calculate_sample_weights 函数 ---
# (因为 FLOS 改回 placeholder，sample_weight 在 model_definition 中计算)
# ---------------------------------------------------------------
