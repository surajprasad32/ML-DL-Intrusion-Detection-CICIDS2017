from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils import info

# Suppress TensorFlow and XGBoost warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ✅ Check GPU availability (for logging)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    info(f"✅ GPU detected: {gpus[0].name}")
else:
    info("⚠️ No GPU detected, using CPU runtime")

def train_random_forest(X_train, y_train):
    info("Training Random Forest (CPU)...")
    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    info("Training XGBoost (GPU accelerated)...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        tree_method='gpu_hist',      # ✅ GPU acceleration
        predictor='gpu_predictor',
        gpu_id=0
    )
    model.fit(X_train, y_train)
    return model

def train_keras_nn(X_train, y_train, input_dim):
    info("Training Neural Network (TensorFlow GPU)...")

    # ✅ Use mixed precision for speed boost on GPU
    if gpus:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid', dtype='float32')  # force output to float32 for metrics
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')]
    )

    # ✅ Add GPU-optimized training parameters
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=1024,
        verbose=1
    )

    info("✅ Neural Network training complete.")
    return model
