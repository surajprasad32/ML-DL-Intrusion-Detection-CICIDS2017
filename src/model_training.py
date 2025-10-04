from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils import info

def train_random_forest(X_train, y_train):
    info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    info("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_keras_nn(X_train, y_train, input_dim):
    info("Training Neural Network...")
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=1024, verbose=1)
    return model
