# -*- coding: utf-8 -*-
import numpy as np, traceback, joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.losses import BinaryCrossentropy, Huber

# --- 位置编码层 ---
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embeddings = layers.Embedding(
            input_dim=self.sequence_length, output_dim=self.output_dim
        )
    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"sequence_length": self.sequence_length, "output_dim": self.output_dim})
        return cfg

# --- DualChannelTransformerModel ---
class DualChannelTransformerModel(keras.Model):
    def __init__(self, num_cir_features=8, num_flos_features=8, model_params=None,
                 learning_rate=1e-4, loss_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.num_cir_features = num_cir_features
        self.num_flos_features = num_flos_features
        self.model_params = model_params if model_params else {}
        self.transformer_layers_count = self.model_params.get('transformer_layers', 2)
        self.attention_heads = self.model_params.get('attention_heads', 4)
        self.embed_dim = self.model_params.get('embed_dim', 64)
        self.ff_dim = self.model_params.get('ff_dim', 128)
        self.dropout_rate = self.model_params.get('dropout_rate', 0.1)
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights if loss_weights else {'nlos_output': 1.0, 'error_output': 1.0}

        if self.embed_dim < self.attention_heads:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be >= attention_heads ({self.attention_heads})")
        if self.embed_dim % self.attention_heads != 0:
            print(f"[Warning] embed_dim ({self.embed_dim}) is not divisible by attention_heads ({self.attention_heads}).")
        self.key_dim_per_head = self.embed_dim // self.attention_heads

        # 消融强度
        self.lambda_mask = 1.0
        self.lambda_ga   = 1.0

        # 内部 scaler
        self.scaler_cir   = StandardScaler()
        self.scaler_flos  = StandardScaler()
        self.scaler_error = StandardScaler()
        self._scalers_fitted = False

        # 训练时使用的类别权重映射（在 train() 中赋值）
        self.class_weight_map = None

        self._build_model_layers()

    def _transformer_encoder(self, name_prefix=""):
        inputs = layers.Input(shape=(None, self.embed_dim))
        attention_output = layers.MultiHeadAttention(
            num_heads=self.attention_heads, key_dim=self.key_dim_per_head, dropout=self.dropout_rate
        )(query=inputs, value=inputs, key=inputs)
        x = layers.Add()([inputs, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_out_norm = x
        ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5)),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim, kernel_regularizer=keras.regularizers.l2(1e-5)),
        ])
        ffn_output = ffn(x)
        x = layers.Add()([attn_out_norm, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return keras.Model(inputs=inputs, outputs=x, name=f"{name_prefix}_encoder_block")

    def _build_model_layers(self):
        self.cir_embedding_pipeline = keras.Sequential([
            layers.Reshape((self.num_cir_features, 1)),
            layers.Dense(self.embed_dim, activation='relu'),
            PositionalEmbedding(self.num_cir_features, self.embed_dim),
            layers.Dropout(self.dropout_rate)
        ], name='cir_embedding_pipeline')

        self.flos_embedding_pipeline = keras.Sequential([
            layers.Reshape((self.num_flos_features, 1)),
            layers.Dense(self.embed_dim, activation='relu'),
            PositionalEmbedding(self.num_flos_features, self.embed_dim),
            layers.Dropout(self.dropout_rate)
        ], name='flos_embedding_pipeline')

        self.cir_encoders  = [self._transformer_encoder(f'cir_encoder_{i+1}')  for i in range(self.transformer_layers_count)]
        self.flos_encoders = [self._transformer_encoder(f'flos_encoder_{i+1}') for i in range(self.transformer_layers_count)]

        self.cir_pooling_for_loss  = layers.GlobalAveragePooling1D(name='cir_pooling_for_loss')
        self.flos_pooling_for_loss = layers.GlobalAveragePooling1D(name='flos_pooling_for_loss')

        self.cross_attention = layers.MultiHeadAttention(num_heads=self.attention_heads, key_dim=self.key_dim_per_head, dropout=self.dropout_rate)
        self.fusion_add  = layers.Add()
        self.fusion_norm = layers.LayerNormalization(epsilon=1e-6)
        self.pooling = layers.GlobalAveragePooling1D()
        self.shared_dense_block = keras.Sequential([
            layers.Dropout(self.dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout_rate)
        ], name='shared_dense_block')
        self.nlos_head  = layers.Dense(1, activation='sigmoid', name='nlos_output')
        self.error_head = layers.Dense(1, activation='linear',  name='error_output')

    def call(self, inputs, training=False):
        cir_input, flos_input = inputs
        cir_embedded  = self.cir_embedding_pipeline(cir_input,  training=training)
        flos_embedded = self.flos_embedding_pipeline(flos_input, training=training)

        cir_encoded = cir_embedded
        for enc in self.cir_encoders:
            cir_encoded = enc(cir_encoded, training=training)
        flos_encoded = flos_embedded
        for enc in self.flos_encoders:
            flos_encoded = enc(flos_encoded, training=training)

        attn = self.cross_attention(query=cir_encoded, value=flos_encoded, key=flos_encoded, training=training)
        fused = self.fusion_add([cir_encoded, attn])
        fused = self.fusion_norm(fused, training=training)

        pooled = self.pooling(fused)
        shared = self.shared_dense_block(pooled, training=training)
        nlos_output  = self.nlos_head(shared)   # (B,1)
        error_output = self.error_head(shared)  # (B,1)

        if training:
            cir_pooled  = self.cir_pooling_for_loss(cir_encoded)
            flos_pooled = self.flos_pooling_for_loss(flos_encoded)
            return nlos_output, error_output, cir_pooled, flos_pooled, shared
        else:
            return nlos_output, error_output

    def compile(self, **kwargs):
        kwargs.pop('loss', None)
        optimizer_config = kwargs.pop('optimizer', "adam")
        super().compile(optimizer=optimizer_config, **kwargs)
        if hasattr(self.optimizer, 'learning_rate'):
            try:
                self.optimizer.learning_rate.assign(self.learning_rate)
            except Exception:
                pass
        self.nlos_loss_fn  = BinaryCrossentropy(name='nlos_loss')
        self.error_loss_fn = Huber(name='error_loss')
        self.nlos_accuracy_metric = keras.metrics.BinaryAccuracy(name='nlos_accuracy')
        self.error_mae_metric     = keras.metrics.MeanAbsoluteError(name='error_mae')

    @property
    def metrics(self):
        return [self.nlos_accuracy_metric, self.error_mae_metric]

    # ---------- 工具函数：安全处理梯度 ----------
    @staticmethod
    def _to_tensor(g):
        if g is None:
            return None
        if isinstance(g, tf.IndexedSlices):
            return tf.convert_to_tensor(g)
        return g

    @staticmethod
    def _dot(g1, g2):
        g1f = tf.reshape(g1, (-1,))
        g2f = tf.reshape(g2, (-1,))
        return tf.reduce_sum(g1f * g2f)

    @staticmethod
    def _add_grads(g1, g2):
        if g1 is None and g2 is None:
            return None
        if g1 is None:
            return DualChannelTransformerModel._to_tensor(g2)
        if g2 is None:
            return DualChannelTransformerModel._to_tensor(g1)
        t1 = DualChannelTransformerModel._to_tensor(g1)
        t2 = DualChannelTransformerModel._to_tensor(g2)
        return t1 + t2

    def train_step(self, data):
        # 我们不再使用来自 fit(...) 的 sample_weight；自己构造 batch 内的类别权重
        if len(data) == 3:
            x, y_true, _ = data
        else:
            x, y_true = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred_nlos, y_pred_error, cir_pooled, flos_pooled, shared_features = self(x, training=True)
            y_true_nlos  = tf.cast(y_true['nlos_output'], tf.float32)   # (B,1) 或 (B,)
            y_true_error = tf.cast(y_true['error_output'], tf.float32)

            # ---- 构造 (B,1) 的样本权重，避免 squeeze 报错 ----
            nlos_sw = None
            if self.class_weight_map:
                w0 = tf.convert_to_tensor(self.class_weight_map.get(0, 1.0), dtype=tf.float32)
                w1 = tf.convert_to_tensor(self.class_weight_map.get(1, 1.0), dtype=tf.float32)
                y_true_nlos_ = tf.reshape(y_true_nlos, (-1, 1))  # 保证 (B,1)
                nlos_sw = y_true_nlos_ * w1 + (1.0 - y_true_nlos_) * w0  # (B,1)

            loss_nlos  = self.nlos_loss_fn(y_true_nlos, y_pred_nlos, sample_weight=nlos_sw)
            loss_error = self.error_loss_fn(y_true_error, y_pred_error)

            main_loss = (self.loss_weights.get('nlos_output', 1.0)  * loss_nlos +
                         self.loss_weights.get('error_output', 1.0) * loss_error)

            # 物理掩码一致性（1 - cos）
            cir_norm  = tf.nn.l2_normalize(cir_pooled,  axis=-1)
            flos_norm = tf.nn.l2_normalize(flos_pooled, axis=-1)
            cosine    = tf.reduce_sum(cir_norm * flos_norm, axis=-1)
            mask_loss = tf.reduce_mean(1.0 - cosine)

            total_loss_wo_ga = main_loss + self.lambda_mask * mask_loss + tf.add_n(self.losses)

        # ---------- PCGrad 融合（IndexedSlices 友好） ----------
        trainable_vars = self.trainable_variables
        grads_nlos  = tape.gradient(loss_nlos,  trainable_vars)
        grads_error = tape.gradient(loss_error, trainable_vars)
        grads_mask  = tape.gradient(total_loss_wo_ga, trainable_vars)

        merged_grads = []
        for g1, g2 in zip(grads_nlos, grads_error):
            if g1 is None and g2 is None:
                merged_grads.append(None); continue
            if g1 is None:
                merged_grads.append(self._to_tensor(g2)); continue
            if g2 is None:
                merged_grads.append(self._to_tensor(g1)); continue
            t1 = self._to_tensor(g1)
            t2 = self._to_tensor(g2)
            dot = self._dot(t1, t2)
            sq2 = self._dot(t2, t2) + 1e-12
            g1_proj = tf.where(dot < 0.0, t1 - (dot / sq2) * t2, t1)
            merged = self.lambda_ga * g1_proj + (1.0 - self.lambda_ga) * t2
            merged_grads.append(merged)

        final_grads = []
        for gm, gm2 in zip(merged_grads, grads_mask):
            final_grads.append(self._add_grads(gm, gm2))

        dense_final_grads = []
        for g, v in zip(final_grads, trainable_vars):
            if g is None:
                dense_final_grads.append(tf.zeros_like(v))
            elif isinstance(g, tf.IndexedSlices):
                dense_final_grads.append(tf.convert_to_tensor(g))
            else:
                dense_final_grads.append(g)

        dense_final_grads, _ = tf.clip_by_global_norm(dense_final_grads, 5.0)
        self.optimizer.apply_gradients(zip(dense_final_grads, trainable_vars))
        del tape

        # 指标
        self.nlos_accuracy_metric.update_state(y_true_nlos, y_pred_nlos, sample_weight=nlos_sw)
        self.error_mae_metric.update_state(y_true_error, y_pred_error)

        # 日志
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': total_loss_wo_ga, 'main_loss': main_loss, 'mask_loss': mask_loss})
        return results

    def test_step(self, data):
        if len(data) == 2:
            x, y_true = data
        else:
            x, y_true, _ = data
        y_pred_nlos, y_pred_error = self(x, training=False)

        y_true_nlos  = tf.cast(y_true['nlos_output'], tf.float32)
        y_true_error = tf.cast(y_true['error_output'], tf.float32)
        loss_nlos  = self.nlos_loss_fn(y_true_nlos,  y_pred_nlos)
        loss_error = self.error_loss_fn(y_true_error, y_pred_error)
        val_loss = (self.loss_weights.get('nlos_output', 1.0) * loss_nlos +
                    self.loss_weights.get('error_output', 1.0) * loss_error)

        self.nlos_accuracy_metric.update_state(y_true_nlos, y_pred_nlos)
        self.error_mae_metric.update_state(y_true_error, y_pred_error)

        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = val_loss
        return results

    def train(self, X_train_cir, X_train_flos, y_train_nlos, y_train_error,
              epochs=10, batch_size=32, class_weight=None,
              validation_data=None, callbacks_list=None):
        # 保存类别权重映射给 train_step 使用
        self.class_weight_map = class_weight if class_weight else None

        # 内部 scaler 拟合
        if not self._scalers_fitted:
            self.scaler_cir.fit(X_train_cir)
            self.scaler_flos.fit(X_train_flos)
            self.scaler_error.fit(y_train_error.reshape(-1, 1))
            self._scalers_fitted = True

        Xc = self.scaler_cir.transform(X_train_cir)
        Xf = self.scaler_flos.transform(X_train_flos)
        ye = self.scaler_error.transform(y_train_error.reshape(-1,1)).ravel()

        keras_inputs  = [Xc, Xf]
        keras_outputs = {'nlos_output': y_train_nlos, 'error_output': ye}

        val_data = None
        if validation_data:
            Xv_c, Xv_f, yv_cls, yv_err = validation_data
            Xv_c = self.scaler_cir.transform(Xv_c)
            Xv_f = self.scaler_flos.transform(Xv_f)
            yv_e = self.scaler_error.transform(yv_err.reshape(-1,1)).ravel()
            val_data = ([Xv_c, Xv_f], {'nlos_output': yv_cls, 'error_output': yv_e})

        # 不再传入 sample_weight，避免 squeeze 形状冲突
        if callbacks_list is None:
            callbacks_list = [
                callbacks.EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=15, restore_best_weights=True, verbose=1),
                callbacks.ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
            ]

        history = super().fit(
            x=keras_inputs, y=keras_outputs,
            epochs=epochs, batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks_list,
            verbose=2
        )
        return history

    def predict(self, X_test_cir, X_test_flos):
        if not self._scalers_fitted:
            print("[错误] 缩放器未拟合，无法预测。"); return None, None
        Xc = self.scaler_cir.transform(X_test_cir)
        Xf = self.scaler_flos.transform(X_test_flos)
        nlos_out, err_out = super().predict([Xc, Xf], verbose=0)
        return nlos_out, err_out

    def save_model(self, model_file_path: Path):
        p = Path(model_file_path)
        print(f"[模型保存] -> {p}")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            super().save(str(p))
            print("[模型保存] 成功")
            return True
        except Exception as e:
            print(f"[错误] 保存失败: {e}")
            traceback.print_exc()
            return False

    def save_scalers(self, cir_scaler_path: Path, flos_scaler_path: Path, error_scaler_path: Path):
        print(f"[Scaler 保存] ...")
        try:
            Path(cir_scaler_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler_cir,   cir_scaler_path)
            joblib.dump(self.scaler_flos,  flos_scaler_path)
            joblib.dump(self.scaler_error, error_scaler_path)
            print(f"[Scaler 保存] 完成")
            return True
        except Exception as e:
            print(f"[错误] 保存 Scaler 失败: {e}")
            return False

    @classmethod
    def load_model_with_scalers(cls, model_path: Path, cir_scaler_path: Path, flos_scaler_path: Path, error_scaler_path: Path):
        print(f"[模型加载] {model_path}")
        try:
            model = models.load_model(
                str(model_path),
                custom_objects={"PositionalEmbedding": PositionalEmbedding, "DualChannelTransformerModel": cls}
            )
            model.scaler_cir   = joblib.load(cir_scaler_path)
            model.scaler_flos  = joblib.load(flos_scaler_path)
            model.scaler_error = joblib.load(error_scaler_path)
            model._scalers_fitted = True
            print("[成功] 模型+Scaler 加载完成")
            return model
        except Exception as e:
            print(f"[错误] 加载失败: {e}")
            traceback.print_exc()
            return None

    def load_scalers(self, cir_scaler_path: Path, flos_scaler_path: Path, error_scaler_path: Path):
        try:
            self.scaler_cir   = joblib.load(cir_scaler_path)
            self.scaler_flos  = joblib.load(flos_scaler_path)
            self.scaler_error = joblib.load(error_scaler_path)
            self._scalers_fitted = True
            print("[Scaler 加载] 完成")
            return True
        except Exception as e:
            print(f"[错误] 加载 Scalers 失败: {e}")
            self._scalers_fitted = False
            return False
