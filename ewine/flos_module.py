# -*- coding: utf-8 -*-
"""
Defines the DualChannelTransformerModel class using TensorFlow/Keras. (增强版)
包含多项架构改进以提升NLOS分类和测距误差缓解性能
"""
import numpy as np
import traceback
import joblib
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.losses import Huber
import tensorflow.keras.backend as K


# 在模块级别定义变量，用于存储训练好的模型和缩放器
gmm_los = None
gmm_nlos = None
scaler_flos = None

# --- 保持原有的位置编码层不变 ---
class PositionalEmbedding(layers.Layer):
    """Adds positional information to the input embeddings."""
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        if sequence_length is None or output_dim is None:
             raise ValueError("PositionalEmbedding requires valid sequence_length and output_dim.")
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embeddings = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.output_dim,
            name=f'{self.name}_embedding'
        )

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# --- 新增：自定义Focal Loss用于处理类别不平衡 ---
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for addressing class imbalance in NLOS classification"""
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        
    def call(self, y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # 计算focal loss
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * self.alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), self.gamma)
        loss = weight * cross_entropy
        return K.mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config

# --- 新增：注意力池化层 ---
class AttentionPooling1D(layers.Layer):
    """Attention-based pooling layer"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        # 计算注意力分数
        scores = K.dot(inputs, self.attention_weights)  # (batch, seq, 1)
        scores = K.softmax(scores, axis=1)
        # 加权求和
        weighted = inputs * scores
        return K.sum(weighted, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class DualChannelTransformerModel:
    """
    增强版双通道Transformer模型
    """
    def __init__(self, num_cir_features=8, num_flos_features=8, model_params=None, 
                 learning_rate=1e-4, loss_weights=None):
        print("\n[模型初始化] === 初始化增强版 DualChannelTransformerModel ===")
        
        if num_cir_features <= 0 or num_flos_features <= 0:
             raise ValueError("Number of CIR and FLOS features must be positive.")
        
        self.num_cir_features = num_cir_features
        self.num_flos_features = num_flos_features
        self.model_params = model_params if model_params else {}
        
        # 获取模型参数
        self.transformer_layers = self.model_params.get('transformer_layers', 2)
        self.attention_heads = self.model_params.get('attention_heads', 4)
        self.embed_dim = self.model_params.get('embed_dim', 64)
        self.ff_dim = self.model_params.get('ff_dim', 128)
        self.dropout_rate = self.model_params.get('dropout_rate', 0.1)
        
        # 验证参数
        if not all(isinstance(p, int) and p > 0 for p in 
                  [self.transformer_layers, self.attention_heads, self.embed_dim, self.ff_dim]):
             raise ValueError("Transformer parameters must be positive integers.")
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError("Dropout rate must be between 0.0 and 1.0.")
            
        self.key_dim_per_head = max(1, self.embed_dim // self.attention_heads)
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights if loss_weights else {'nlos_output': 1.0, 'error_output': 1.0}
        
        self.model = None
        self.history = None
        self.scaler_cir = StandardScaler()
        self.scaler_flos = StandardScaler()
        self.scaler_error = StandardScaler()
        self._scalers_fitted = False
        
        print("[模型初始化] 增强版模型初始化完成")

    def _transformer_encoder_enhanced(self, embed_dim, num_heads, key_dim_per_head, 
                                     ff_dim, dropout_rate, name_prefix=""):
        """创建增强版的Transformer Encoder块，添加Layer Normalization在前"""
        inputs = layers.Input(shape=(None, embed_dim), name=f"{name_prefix}_input")
        
        # Pre-normalization (更稳定的训练)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_pre_norm1")(inputs)
        
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim_per_head, 
            dropout=dropout_rate,
            name=f"{name_prefix}_mha"
        )(query=x, value=x, key=x)
        
        # Residual connection
        x = layers.Add(name=f"{name_prefix}_add1")([inputs, attention_output])
        
        # Pre-normalization for FFN
        attn_output = x
        x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_pre_norm2")(x)
        
        # Enhanced Feed Forward Network with GLU (Gated Linear Unit)
        ffn_gate = layers.Dense(ff_dim, activation="sigmoid", name=f"{name_prefix}_ffn_gate")(x)
        ffn_value = layers.Dense(ff_dim, activation="relu", name=f"{name_prefix}_ffn_value")(x)
        x = layers.Multiply(name=f"{name_prefix}_glu")([ffn_gate, ffn_value])
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_ffn_dropout")(x)
        x = layers.Dense(embed_dim, name=f"{name_prefix}_ffn_project")(x)
        
        # Residual connection
        x = layers.Add(name=f"{name_prefix}_add2")([attn_output, x])
        
        return keras.Model(inputs=inputs, outputs=x, name=f"{name_prefix}_encoder_block")

    def build_model(self):
        """构建增强版模型架构"""
        if self.model is not None:
            return

        print("[模型构建] --- 开始构建增强版双通道模型 ---")
        
        embed_dim = self.embed_dim
        num_heads = self.attention_heads
        key_dim_per_head = self.key_dim_per_head
        ff_dim = self.ff_dim
        dropout_rate = self.dropout_rate
        num_encoder_layers = self.transformer_layers

        # === 输入层 ===
        cir_input = layers.Input(shape=(self.num_cir_features,), name='cir_input')
        flos_input = layers.Input(shape=(self.num_flos_features,), name='flos_input')

        # === CIR通道处理 ===
        cir_reshaped = layers.Reshape((self.num_cir_features, 1), name='cir_reshape')(cir_input)
        # 使用更复杂的嵌入层
        cir_embedded = layers.Dense(embed_dim, name='cir_embedding_1')(cir_reshaped)
        cir_embedded = layers.LayerNormalization(name='cir_embed_norm')(cir_embedded)
        cir_embedded = layers.Activation('gelu', name='cir_embed_activation')(cir_embedded)  # 使用GELU激活
        cir_pos_encoded = PositionalEmbedding(self.num_cir_features, embed_dim, 
                                              name='cir_pos_embedding')(cir_embedded)
        cir_encoder_input = layers.Dropout(dropout_rate, name='cir_input_dropout')(cir_pos_encoded)

        # === FLOS通道处理 ===
        flos_reshaped = layers.Reshape((self.num_flos_features, 1), name='flos_reshape')(flos_input)
        flos_embedded = layers.Dense(embed_dim, name='flos_embedding_1')(flos_reshaped)
        flos_embedded = layers.LayerNormalization(name='flos_embed_norm')(flos_embedded)
        flos_embedded = layers.Activation('gelu', name='flos_embed_activation')(flos_embedded)
        flos_pos_encoded = PositionalEmbedding(self.num_flos_features, embed_dim, 
                                               name='flos_pos_embedding')(flos_embedded)
        flos_encoder_input = layers.Dropout(dropout_rate, name='flos_input_dropout')(flos_pos_encoded)

        # === 独立的Transformer编码器 (使用增强版) ===
        cir_encoded = cir_encoder_input
        flos_encoded = flos_encoder_input
        
        for i in range(num_encoder_layers):
            cir_encoder_layer = self._transformer_encoder_enhanced(
                embed_dim, num_heads, key_dim_per_head, ff_dim, dropout_rate, 
                name_prefix=f'cir_encoder_{i+1}'
            )
            flos_encoder_layer = self._transformer_encoder_enhanced(
                embed_dim, num_heads, key_dim_per_head, ff_dim, dropout_rate, 
                name_prefix=f'flos_encoder_{i+1}'
            )
            cir_encoded = cir_encoder_layer(cir_encoded)
            flos_encoded = flos_encoder_layer(flos_encoded)

        # === 双向Cross-Attention融合 ===
        # CIR query, FLOS key/value
        cross_attention_cf = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim_per_head, dropout=dropout_rate, 
            name='cross_attention_cir_flos'
        )(query=cir_encoded, value=flos_encoded, key=flos_encoded)
        
        # FLOS query, CIR key/value
        cross_attention_fc = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim_per_head, dropout=dropout_rate, 
            name='cross_attention_flos_cir'
        )(query=flos_encoded, value=cir_encoded, key=cir_encoded)
        
        # 融合两个方向的attention
        fused_cf = layers.Add(name='cross_attention_add_cf')([cir_encoded, cross_attention_cf])
        fused_cf = layers.LayerNormalization(epsilon=1e-6, name='cross_attention_norm_cf')(fused_cf)
        
        fused_fc = layers.Add(name='cross_attention_add_fc')([flos_encoded, cross_attention_fc])
        fused_fc = layers.LayerNormalization(epsilon=1e-6, name='cross_attention_norm_fc')(fused_fc)
        
        # 连接两个方向的特征
        fused_features = layers.Concatenate(axis=1, name='bidirectional_concat')([fused_cf, fused_fc])
        
        # === 多尺度池化 ===
        # 1. 全局平均池化
        pool_avg = layers.GlobalAveragePooling1D(name='global_avg_pooling')(fused_features)
        
        # 2. 全局最大池化
        pool_max = layers.GlobalMaxPooling1D(name='global_max_pooling')(fused_features)
        
        # 3. 注意力池化
        pool_attention = AttentionPooling1D(name='attention_pooling')(fused_features)
        
        # 4. 第一个和最后一个时间步的特征
        first_step = layers.Lambda(lambda x: x[:, 0, :], name='first_step')(fused_features)
        last_step = layers.Lambda(lambda x: x[:, -1, :], name='last_step')(fused_features)
        
        # 连接所有池化特征
        pooled_features = layers.Concatenate(name='multi_scale_concat')(
            [pool_avg, pool_max, pool_attention, first_step, last_step]
        )
        pooled_features = layers.Dropout(dropout_rate, name='pool_dropout')(pooled_features)

        # === 深度共享网络 (带批归一化和残差连接) ===
        # 第一层
        shared_1 = layers.Dense(512, name='shared_dense_1')(pooled_features)
        shared_1 = layers.BatchNormalization(name='bn_1')(shared_1)
        shared_1 = layers.Activation('gelu')(shared_1)
        shared_1 = layers.Dropout(dropout_rate)(shared_1)
        
        # 第二层 (带残差)
        shared_2 = layers.Dense(256, name='shared_dense_2')(shared_1)
        shared_2 = layers.BatchNormalization(name='bn_2')(shared_2)
        shared_2 = layers.Activation('gelu')(shared_2)
        shared_2 = layers.Dropout(dropout_rate)(shared_2)
        
        # 第三层
        shared_3 = layers.Dense(128, name='shared_dense_3')(shared_2)
        shared_3 = layers.BatchNormalization(name='bn_3')(shared_3)
        shared_3 = layers.Activation('gelu')(shared_3)
        final_shared = layers.Dropout(dropout_rate)(shared_3)

        # === 任务特定的输出头 ===
        # NLOS分类头 (更深的网络)
        nlos_branch = layers.Dense(64, activation='relu', name='nlos_branch_1')(final_shared)
        nlos_branch = layers.BatchNormalization(name='nlos_bn')(nlos_branch)
        nlos_branch = layers.Dropout(dropout_rate/2)(nlos_branch)
        nlos_branch = layers.Dense(32, activation='relu', name='nlos_branch_2')(nlos_branch)
        nlos_output = layers.Dense(1, activation='sigmoid', name='nlos_output')(nlos_branch)
        
        # 误差回归头 (更深的网络)
        error_branch = layers.Dense(128, activation='relu', name='error_branch_1')(final_shared)
        error_branch = layers.BatchNormalization(name='error_bn')(error_branch)
        error_branch = layers.Dropout(dropout_rate/2)(error_branch)
        error_branch = layers.Dense(64, activation='relu', name='error_branch_2')(error_branch)
        error_branch = layers.Dense(32, activation='relu', name='error_branch_3')(error_branch)
        error_output = layers.Dense(1, activation='linear', name='error_output')(error_branch)

        # === 创建模型 ===
        self.model = models.Model(
            inputs=[cir_input, flos_input], 
            outputs=[nlos_output, error_output], 
            name="EnhancedDualChannelTransformer"
        )

        # === 编译模型 (使用改进的损失函数和优化器) ===
        # 使用AdamW优化器 (带权重衰减)
        optimizer = optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=1e-5,  # L2正则化
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # 创建Focal Loss实例
        focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'nlos_output': focal_loss,  # 使用Focal Loss
                'error_output': Huber(delta=1.0)  # Huber loss对异常值更鲁棒
            },
            loss_weights=self.loss_weights,
            metrics={
                'nlos_output': [
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc', curve='ROC'),
                    tf.keras.metrics.AUC(name='pr_auc', curve='PR')
                ],
                'error_output': [
                    tf.keras.metrics.MeanAbsoluteError(name='mae'),
                    tf.keras.metrics.MeanSquaredError(name='mse'),
                    tf.keras.metrics.RootMeanSquaredError(name='rmse')
                ]
            }
        )
        
        print("[模型构建] 增强版模型构建和编译完成")
        print(f"[模型构建] 总参数量: {self.model.count_params():,}")
        self.model.summary(line_length=120)

    def train(self, X_train_cir, X_train_flos, y_train_nlos, y_train_error,
              epochs=10, batch_size=32, class_weight=None,
              validation_data=None, callbacks_list=None):
        """
        训练模型，包含增强的回调函数和训练策略
        """
        print("\n[模型训练] === 开始增强版模型训练 ===")
        
        if self.model is None:
            self.build_model()
        if self.model is None:
            print("[错误] 模型构建失败")
            return False

        # 拟合缩放器
        if not self._scalers_fitted:
            print("[模型训练] 正在拟合缩放器...")
            try:
                X_train_cir_scaled = self.scaler_cir.fit_transform(X_train_cir)
                X_train_flos_scaled = self.scaler_flos.fit_transform(X_train_flos)
                y_train_error_reshaped = y_train_error.reshape(-1, 1) if y_train_error.ndim == 1 else y_train_error
                y_train_error_scaled = self.scaler_error.fit_transform(y_train_error_reshaped)
                self._scalers_fitted = True
                print("[模型训练] 缩放器拟合完成")
            except Exception as e:
                print(f"[错误] 缩放器拟合失败: {e}")
                return False
        else:
            print("[模型训练] 使用已拟合的缩放器...")
            try:
                X_train_cir_scaled = self.scaler_cir.transform(X_train_cir)
                X_train_flos_scaled = self.scaler_flos.transform(X_train_flos)
                y_train_error_reshaped = y_train_error.reshape(-1, 1) if y_train_error.ndim == 1 else y_train_error
                y_train_error_scaled = self.scaler_error.transform(y_train_error_reshaped)
            except Exception as e:
                print(f"[错误] 数据缩放失败: {e}")
                return False

        # 准备输入输出
        keras_inputs = [X_train_cir_scaled, X_train_flos_scaled]
        keras_outputs = {'nlos_output': y_train_nlos, 'error_output': y_train_error_scaled.flatten()}

        # 处理验证数据
        keras_validation_data = None
        if validation_data:
            print("[模型训练] 准备验证数据...")
            try:
                X_val_cir, X_val_flos, y_val_nlos, y_val_error = validation_data
                X_val_cir_scaled = self.scaler_cir.transform(X_val_cir)
                X_val_flos_scaled = self.scaler_flos.transform(X_val_flos)
                y_val_error_reshaped = y_val_error.reshape(-1, 1) if y_val_error.ndim == 1 else y_val_error
                y_val_error_scaled = self.scaler_error.transform(y_val_error_reshaped)
                
                keras_validation_data = (
                    [X_val_cir_scaled, X_val_flos_scaled],
                    {'nlos_output': y_val_nlos, 'error_output': y_val_error_scaled.flatten()}
                )
                print("[模型训练] 验证数据准备完成")
            except Exception as e:
                print(f"[警告] 验证数据处理失败: {e}")
                keras_validation_data = None

        # 样本权重
        keras_sample_weight = None
        if class_weight:
            try:
                sample_weights_nlos = np.array([class_weight.get(label, 1.0) for label in y_train_nlos])
                keras_sample_weight = {'nlos_output': sample_weights_nlos}
            except Exception as e:
                print(f"[警告] 样本权重计算失败: {e}")

        # === 增强的回调函数 ===
        if callbacks_list is None:
            # 学习率预热函数
            def lr_schedule(epoch):
                warmup_epochs = 10
                if epoch < warmup_epochs:
                    return self.learning_rate * (epoch + 1) / warmup_epochs
                else:
                    # Cosine annealing
                    progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                    return self.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
            
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss' if keras_validation_data else 'loss',
                    patience=30,  # 增加耐心值
                    restore_best_weights=True,
                    verbose=1,
                    mode='min'
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if keras_validation_data else 'loss',
                    factor=0.5,
                    patience=15,
                    min_lr=1e-7,
                    verbose=1,
                    mode='min'
                ),
                callbacks.LearningRateScheduler(lr_schedule, verbose=0),
                callbacks.ModelCheckpoint(
                    filepath='best_dual_transformer_enhanced.keras',
                    monitor='val_loss' if keras_validation_data else 'loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                    mode='min'
                ),
                # 新增：TensorBoard日志
                callbacks.TensorBoard(
                    log_dir='./logs/dual_transformer',
                    histogram_freq=1,
                    write_graph=True,
                    update_freq='epoch'
                )
            ]
            print("[模型训练] 使用增强版回调函数")

        print(f"[模型训练] 开始训练，共 {epochs} 轮，批大小 {batch_size}")

        try:
            self.history = self.model.fit(
                x=keras_inputs,
                y=keras_outputs,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=keras_validation_data,
                sample_weight=keras_sample_weight,
                callbacks=callbacks_list,
                verbose=1,  # 显示详细进度
                shuffle=True  # 每个epoch打乱数据
            )
            print("[模型训练] 训练完成")
            return True
        except Exception as e:
            print(f"[错误] 训练过程出错: {e}")
            traceback.print_exc()
            return False

    # 保持原有的predict, save_model, save_scalers等方法不变
    def predict(self, X_test_cir, X_test_flos):
        """预测函数保持不变"""
        if self.model is None:
            print("[错误] 模型未训练")
            return None, None
        if not self._scalers_fitted:
            print("[错误] 缩放器未拟合")
            return None, None

        try:
            X_test_cir_scaled = self.scaler_cir.transform(X_test_cir)
            X_test_flos_scaled = self.scaler_flos.transform(X_test_flos)
        except Exception as e:
            print(f"[错误] 测试数据缩放失败: {e}")
            return None, None

        try:
            keras_test_inputs = [X_test_cir_scaled, X_test_flos_scaled]
            predictions = self.model.predict(keras_test_inputs)
            y_pred_nlos_prob = predictions[0]
            y_pred_error_scaled = predictions[1]
        except Exception as e:
            print(f"[错误] 预测失败: {e}")
            return None, None

        return y_pred_nlos_prob, y_pred_error_scaled

    # 其余方法保持不变...
    def save_model(self, model_file_path: Path):
        """保存模型"""
        model_file_path = Path(model_file_path)
        print(f"[模型保存] 保存模型到: {model_file_path.name}")
        if self.model is None:
            print("[警告] 模型不存在")
            return False
        try:
            model_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(model_file_path))
            print("[模型保存] 保存成功")
            return True
        except Exception as e:
            print(f"[错误] 保存失败: {e}")
            return False

    def save_scalers(self, cir_scaler_path: Path, flos_scaler_path: Path, error_scaler_path: Path):
        """保存缩放器"""
        # 保持原有实现
        cir_scaler_path = Path(cir_scaler_path)
        flos_scaler_path = Path(flos_scaler_path)
        error_scaler_path = Path(error_scaler_path)
        print(f"[Scaler 保存] 保存缩放器...")
        
        try:
            cir_scaler_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[错误] 创建目录失败: {e}")
            return False

        saved_any = False
        try:
            joblib.dump(self.scaler_cir, cir_scaler_path)
            print(f"   - CIR Scaler: {cir_scaler_path.name}")
            saved_any = True
        except Exception as e:
            print(f"[错误] CIR Scaler保存失败: {e}")

        try:
            joblib.dump(self.scaler_flos, flos_scaler_path)
            print(f"   - FLOS Scaler: {flos_scaler_path.name}")
            saved_any = True
        except Exception as e:
            print(f"[错误] FLOS Scaler保存失败: {e}")

        try:
            joblib.dump(self.scaler_error, error_scaler_path)
            print(f"   - Error Scaler: {error_scaler_path.name}")
            saved_any = True
        except Exception as e:
            print(f"[错误] Error Scaler保存失败: {e}")

        return saved_any

def _find_best_gmm(X, max_components, covariance_type, random_state):
    """
    一个辅助函数，用于通过BIC准则寻找最佳的GMM成分数。
    """
    if X.shape[0] < max_components:
        max_components = X.shape[0]
        
    lowest_bic = np.inf
    best_gmm = None
    
    print(f"    -> 正在为 {X.shape[0]} 个样本搜索最佳GMM (1至{max_components}个成分)...")
    
    for n_components in range(1, max_components + 1):
        try:
            # 现在 GaussianMixture 已经被正确导入，可以正常使用
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type=covariance_type,
                                  random_state=random_state,
                                  n_init=5,
                                  max_iter=200,
                                  reg_covar=1e-6)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
        except Exception as e:
            print(f"       [警告] GMM在{n_components}个成分时拟合失败: {e}")
            continue
            
    if best_gmm:
        print(f"    -> 最佳GMM已找到，成分数: {best_gmm.n_components}")    
    else:
        print(f"    -> [错误] 未能找到有效的GMM模型。")
        
    return best_gmm

def fit_flos_gmm_models(X_train_cir, y_train_nlos, 
                        max_components_search=30,  # <-- 根据您的日志，上限设为30
                        covariance_type='diag', 
                        random_state=42):
    """
    为LOS和NLOS数据分别训练并存储最佳的GMM模型。
    """
    global gmm_los, gmm_nlos, scaler_flos
    
    print("\n--- [FLOS GMM] 开始拟合LOS和NLOS的GMM模型 ---")
    
    X_los = X_train_cir[y_train_nlos == 0]
    X_nlos = X_train_cir[y_train_nlos == 1]
    
    if X_los.shape[0] == 0 or X_nlos.shape[0] == 0:
        print("[错误] 训练数据中缺少LOS或NLOS样本，无法拟合GMM。")
        return False
        
    print("[FLOS GMM] 正在为FLOS特征拟合StandardScaler...")
    scaler_flos = StandardScaler()
    scaler_flos.fit(X_train_cir)
    print("[FLOS GMM] StandardScaler拟合完成。")
    
    X_los_scaled = scaler_flos.transform(X_los)
    X_nlos_scaled = scaler_flos.transform(X_nlos)
    
    print("\n[FLOS GMM] 处理LOS数据...")
    gmm_los = _find_best_gmm(X_los_scaled, max_components_search, covariance_type, random_state)
    
    print("\n[FLOS GMM] 处理NLOS数据...")
    gmm_nlos = _find_best_gmm(X_nlos_scaled, max_components_search, covariance_type, random_state)
    
    if gmm_los is not None and gmm_nlos is not None:
        print("\n--- [FLOS G-MM] LOS和NLOS的GMM模型均已成功拟合。---")
        return True
    else:
        print("\n--- [错误] GMM模型拟合失败。---")
        return False
def save_flos_gmm_components(save_dir, scaler_filename, gmm_los_filename, gmm_nlos_filename):
    """
    将训练好的FLOS GMM组件（scaler, gmm_los, gmm_nlos）保存到文件。
    """
    global gmm_los, gmm_nlos, scaler_flos
    
    print("\n--- [FLOS GMM] 开始保存GMM模型和缩放器 ---")
    
    # 检查模型和缩放器是否已成功拟合
    if scaler_flos is None or gmm_los is None or gmm_nlos is None:
        print("[错误] Scaler或GMM模型尚未拟合，无法保存。")
        return

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 保存Scaler
        scaler_path = save_dir_path / scaler_filename
        joblib.dump(scaler_flos, scaler_path)
        print(f"    -> StandardScaler 已保存至: {scaler_path.name}")
        
        # 保存GMM for LOS
        gmm_los_path = save_dir_path / gmm_los_filename
        joblib.dump(gmm_los, gmm_los_path)
        print(f"    -> GMM (LOS) 模型已保存至: {gmm_los_path.name}")
        
        # 保存GMM for NLOS
        gmm_nlos_path = save_dir_path / gmm_nlos_filename
        joblib.dump(gmm_nlos, gmm_nlos_path)
        print(f"    -> GMM (NLOS) 模型已保存至: {gmm_nlos_path.name}")
        
        print("--- [FLOS GMM] 所有组件保存成功。 ---")
        
    except Exception as e:
        print(f"[错误] 保存FLOS GMM组件时发生错误: {e}")

def calculate_flos_features_gmm(X_cir):
    """
    使用已拟合的GMM模型为输入的CIR数据计算FLOS特征。
    FLOS特征是数据点在LOS GMM和NLOS GMM下的对数似然得分。
    """
    global gmm_los, gmm_nlos, scaler_flos
    
    # 检查模型和缩放器是否已成功拟合
    if scaler_flos is None or gmm_los is None or gmm_nlos is None:
        print("[错误] Scaler或GMM模型尚未拟合，无法计算FLOS特征。")
        return None

    print(f"--- [FLOS GMM] 正在为 {X_cir.shape[0]} 个样本计算FLOS特征...")
    
    try:
        # 1. 使用已拟合的缩放器对输入数据进行标准化
        X_cir_scaled = scaler_flos.transform(X_cir)
        
        # 2. 计算每个样本在LOS GMM下的对数似然
        log_likelihood_los = gmm_los.score_samples(X_cir_scaled)
        
        # 3. 计算每个样本在NLOS GMM下的对数似然
        log_likelihood_nlos = gmm_nlos.score_samples(X_cir_scaled)
        
        # 4. 将两个对数似然值合并为新的特征数组
        # 新特征的维度将是 (样本数, 2)
        flos_features = np.column_stack((log_likelihood_los, log_likelihood_nlos))
        
        print("--- [FLOS GMM] FLOS特征计算完成。---")
        
        return flos_features
        
    except Exception as e:
        print(f"[错误] 计算FLOS特征时发生错误: {e}")
        return None