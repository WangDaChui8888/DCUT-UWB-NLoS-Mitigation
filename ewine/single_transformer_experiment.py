# -*- coding: utf-8 -*-
"""
Single Channel Transformer 对比实验模块
适配 K 折交叉验证：在指定的训练/验证折叠上训练和预测。
"""
import numpy as np
import pandas as pd
# import joblib # No longer needed here
import time
import traceback
# from pathlib import Path # No longer needed here
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError # To check scaler status

# 假设 evaluation_utils 和 config 在可导入路径下
try:
    from evaluation_utils import evaluate_classification, evaluate_regression
    import config # Needed for default filenames if function call omits them
except ImportError:
    print("[警告] single_transformer_experiment.py: 无法导入 'evaluation_utils' 或 'config'。")
    # Define dummy functions/variables if needed
    def evaluate_classification(*args, **kwargs): return {}
    def evaluate_regression(*args, **kwargs): return {}
    class Config:
        SINGLE_TRANSFORMER_MODEL_FILE = 'single_transformer_model_combined.keras'; SINGLE_TRANSFORMER_RESULTS_CSV = 'single_transformer_predictions_combined.csv'; SINGLE_TRANSFORMER_RESULTS_MAT = 'single_transformer_results_combined.mat'
    config = Config()

# 从 model_definition 导入 PositionalEmbedding
try:
    from model_definition import PositionalEmbedding
except ImportError:
    print("[错误] single_transformer_experiment.py: 无法从 'model_definition' 导入 'PositionalEmbedding'。请确保文件存在且路径正确。")
    # Provide a fallback definition if necessary for the script to be syntactically correct
    class PositionalEmbedding(layers.Layer):
        def __init__(self, sequence_length, output_dim, **kwargs): super().__init__(**kwargs); self.sequence_length = sequence_length; self.output_dim = output_dim; self.position_embeddings = layers.Embedding(input_dim=self.sequence_length, output_dim=self.output_dim)
        def call(self, inputs): length = tf.shape(inputs)[1]; positions = tf.range(start=0, limit=length, delta=1); embedded_positions = self.position_embeddings(positions); return inputs + embedded_positions
        def compute_mask(self, inputs, mask=None): return mask
        def get_config(self): config = super().get_config(); config.update({"sequence_length": self.sequence_length, "output_dim": self.output_dim}); return config


def _transformer_encoder(embed_dim, num_heads, ff_dim, dropout_rate, name_prefix=""):
    """创建一个 Transformer Encoder 层 (Self-Attention -> FF). (保持不变)"""
    key_dim_head = max(1, embed_dim // num_heads) # 确保 key_dim >= 1
    inputs = layers.Input(shape=(None, embed_dim)) # (Batch, Seq, Dim)
    # Layer 1: Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim_head, dropout=dropout_rate,
        name=f"{name_prefix}_mha"
    )(query=inputs, value=inputs, key=inputs) # Self-attention
    x = layers.Add(name=f"{name_prefix}_add1")([inputs, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_norm1")(x)
    attn_out_norm = x # Store for residual connection
    # Layer 2: Feed Forward Network
    ffn = keras.Sequential(
        [
            layers.Dense(ff_dim, activation="relu", name=f"{name_prefix}_ffn_dense1"),
            layers.Dropout(dropout_rate, name=f"{name_prefix}_ffn_dropout"), # 在 FFN 中也加 Dropout
            layers.Dense(embed_dim, name=f"{name_prefix}_ffn_dense2"),
        ],
        name=f"{name_prefix}_ffn"
    )
    x_ffn = ffn(x)
    x = layers.Add(name=f"{name_prefix}_add2")([attn_out_norm, x_ffn]) # Add to output of first normalization
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_norm2")(x)
    return keras.Model(inputs=inputs, outputs=x, name=f"{name_prefix}_encoder_block")

def build_single_transformer_model(num_cir_features, transformer_params, learning_rate, loss_weights):
    """构建单通道 Transformer 模型 (保持不变)"""
    embed_dim = transformer_params.get('embed_dim', 64)
    num_heads = transformer_params.get('attention_heads', 4)
    ff_dim = transformer_params.get('ff_dim', 128)
    dropout_rate = transformer_params.get('dropout_rate', 0.1)
    num_encoder_layers = transformer_params.get('transformer_layers', 2)

    # 输入层
    cir_input = layers.Input(shape=(num_cir_features,), name='cir_input') # (None, num_features)

    # 嵌入 + 位置编码
    cir_reshaped = layers.Reshape((num_cir_features, 1))(cir_input)
    cir_embedded = layers.Dense(embed_dim, name='cir_embedding')(cir_reshaped)
    cir_pos_encoded = PositionalEmbedding(num_cir_features, embed_dim, name='cir_pos_embedding')(cir_embedded)
    encoder_input = layers.Dropout(dropout_rate, name='input_dropout')(cir_pos_encoded)

    # Transformer Encoder 块
    encoded_features = encoder_input
    for i in range(num_encoder_layers):
        encoder_layer = _transformer_encoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name_prefix=f"transformer_layer_{i+1}"
        )
        encoded_features = encoder_layer(encoded_features)

    # 输出处理
    pooled_features = layers.GlobalAveragePooling1D(name='global_avg_pooling')(encoded_features)
    pooled_features = layers.Dropout(dropout_rate)(pooled_features) # Dropout before dense

    shared_dense = layers.Dense(64, activation='relu', name='shared_dense_1')(pooled_features)
    shared_dense = layers.Dropout(dropout_rate)(shared_dense)

    # 输出头
    nlos_output = layers.Dense(1, activation='sigmoid', name='nlos_output')(shared_dense)
    error_output = layers.Dense(1, activation='linear', name='error_output')(shared_dense)

    model = models.Model(inputs=cir_input, outputs=[nlos_output, error_output], name="SingleChannelTransformer")

    # 编译
    optimizer = optimizers.Adam(learning_rate=learning_rate) # Use standard Adam
    model.compile(optimizer=optimizer,
                  loss={'nlos_output': 'binary_crossentropy', 'error_output': Huber()},
                  loss_weights=loss_weights,
                  metrics={'nlos_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]})
    print("[Single Transformer 模型] 模型构建和编译完成。")
    model.summary()
    return model

# --- 修改后的函数，用于处理单个 K 折 ---
def run_single_transformer_fold_experiment(
    # --- 训练数据 (当前折叠, 已缩放) ---
    X_train_cir_scaled,   # 缩放后的训练 CIR 特征
    y_train_nlos,         # 训练 NLOS 标签
    y_train_error_scaled, # 缩放后的训练 Error 目标
    # --- 验证数据 (当前折叠, 已缩放) ---
    X_val_cir_scaled,     # 缩放后的验证 CIR 特征
    y_val_nlos,           # 验证 NLOS 标签
    y_val_error_scaled,   # 缩放后的验证 Error 目标
    # --- 逆缩放器 ---
    scaler_error,         # 拟合好的 Error 缩放器
    # --- 模型和训练参数 ---
    transformer_params: dict,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loss_weights: dict,
    # --- 其他 ---
    fold_number: int,      # 当前折叠编号
    class_weights_dict=None # 类别权重
    ):
    """
    在 K 折交叉验证的单个折叠上运行单通道 Transformer 实验。
    训练模型并在验证集上进行预测。

    Args:
        (与 dnn_experiment 类似, 但使用 transformer_params)
        ...

    Returns:
        tuple: (trained_model, history, train_time, val_predictions, val_probabilities, pred_time_val)
               如果失败则返回 (None, None, -1, None, None, -1)
        (格式与 dnn_experiment 返回值一致)
    """
    print(f"\n[Single Transformer 实验 - 折叠 {fold_number}] === 开始运行 ===")
    model = None
    history = None
    train_time = -1
    pred_time_val = -1
    y_pred_nlos_val = None
    y_pred_error_val_orig = None
    y_pred_nlos_prob_val = None

    try:
        # --- 数据形状检查 (可选) ---
        print(f"[ST 折叠 {fold_number}] 训练数据形状: CIR={X_train_cir_scaled.shape}, NLOS={y_train_nlos.shape}, Error={y_train_error_scaled.shape}")
        print(f"[ST 折叠 {fold_number}] 验证数据形状: CIR={X_val_cir_scaled.shape}, NLOS={y_val_nlos.shape}, Error={y_val_error_scaled.shape}")

        # --- 模型构建 ---
        num_cir_features = X_train_cir_scaled.shape[1]
        model = build_single_transformer_model(num_cir_features, transformer_params, learning_rate, loss_weights)

        # --- 回调函数 ---
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
        ]

        # --- 样本权重 ---
        keras_sample_weight = None
        if class_weights_dict:
            try:
                sample_weights_nlos = np.array([class_weights_dict.get(label, 1.0) for label in y_train_nlos])
                keras_sample_weight = {'nlos_output': sample_weights_nlos}
                print(f"[ST 折叠 {fold_number}] 将应用样本权重 (仅用于 NLOS 输出)。")
            except Exception as e: print(f"[警告 - 折叠 {fold_number}] ST 计算样本权重时出错: {e}")

        # --- 模型训练 ---
        print(f"[ST 折叠 {fold_number}] 开始训练，共 {epochs} 轮...")
        train_start_time = time.time()
        # 准备验证数据格式
        validation_data_prepared = (
            X_val_cir_scaled,
            {'nlos_output': y_val_nlos, 'error_output': y_val_error_scaled}
        )

        history_obj = model.fit(
            x=X_train_cir_scaled, # 单一输入
            y={'nlos_output': y_train_nlos, 'error_output': y_train_error_scaled},
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data_prepared, # 使用显式验证集
            callbacks=callbacks_list,
            sample_weight=keras_sample_weight,
            verbose=2
        )
        train_time = time.time() - train_start_time
        history = history_obj.history
        print(f"[ST 折叠 {fold_number}] 训练完成，耗时: {train_time:.2f} 秒")

        # --- 在验证集上预测 ---
        print(f"[ST 折叠 {fold_number}] 在验证集上进行预测...")
        pred_start_time = time.time()
        predictions_val = model.predict(X_val_cir_scaled)
        pred_time_val = time.time() - pred_start_time
        print(f"[ST 折叠 {fold_number}] 验证集预测完成，耗时: {pred_time_val:.4f} 秒")

        y_pred_nlos_prob_val = predictions_val[0].flatten()
        y_pred_error_scaled_val = predictions_val[1]

        # --- 后处理验证集预测 ---
        y_pred_nlos_val = (y_pred_nlos_prob_val > 0.5).astype(int)
        print(f"[ST 折叠 {fold_number}] 逆缩放验证集预测误差...")
        try:
            # Check if scaler is fitted
            if not hasattr(scaler_error, 'mean_') or scaler_error.mean_ is None:
                 raise NotFittedError("Error scaler is not fitted.")
            y_pred_error_val_orig = scaler_error.inverse_transform(y_pred_error_scaled_val.reshape(-1, 1)).flatten()
        except NotFittedError as nfe:
             print(f"[错误 - 折叠 {fold_number}] 无法逆缩放误差: {nfe}")
             y_pred_error_val_orig = np.full_like(y_pred_error_scaled_val, np.nan) # Return NaN on failure
        except Exception as e_inv:
            print(f"[错误 - 折叠 {fold_number}] 逆缩放验证集误差时出错: {e_inv}")
            traceback.print_exc()
            y_pred_error_val_orig = np.full_like(y_pred_error_scaled_val, np.nan) # Return NaN on failure

        # --- 不在此处进行评估或保存 ---

    except Exception as e:
        print(f"[错误 - 折叠 {fold_number}] Single Transformer 实验执行失败: {e}")
        traceback.print_exc()
        tf.keras.backend.clear_session() # Clear session on error
        return None, None, -1, None, None, -1 # Return failure

    print(f"[Single Transformer 实验 - 折叠 {fold_number}] === 运行结束 ===")
    # 不清除会话，模型需要返回
    # tf.keras.backend.clear_session()

    return model, history, train_time, (y_pred_nlos_val, y_pred_error_val_orig), y_pred_nlos_prob_val, pred_time_val
