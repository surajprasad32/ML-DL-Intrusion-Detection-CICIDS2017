import os
import joblib
from src.data_preprocessing import load_cicids_data, preprocess_data
from src.model_training import train_xgboost, train_keras_nn
from src.evaluation import evaluate_model
from src.utils import info
import tensorflow as tf
import json

DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"

def main():
    info("Starting GPU-based Intrusion Detection Training (XGBoost + Neural Network)")

    # ✅ GPU check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        info(f"GPU detected: {gpus[0].name}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            pass
    else:
        info("No GPU detected, running on CPU!")

    # ===============================================
    # Load and preprocess data
    # ===============================================
    data = load_cicids_data(DATA_DIR)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # ===============================================
    # Train models
    # ===============================================
    info(">>> Training XGBoost Model (GPU)...")
    xgb_model = train_xgboost(X_train, y_train)

    info(">>> Training Neural Network Model (GPU)...")
    nn_model = train_keras_nn(X_train, y_train, X_train.shape[1])

    # ===============================================
    # Evaluate models
    # ===============================================
    info(">>> Evaluating Models...")
    results = {}
    results["XGBoost"] = evaluate_model("XGBoost", xgb_model, X_test, y_test)
    results["NeuralNet"] = evaluate_model("Neural Network", nn_model, X_test, y_test, is_keras=True)

    # ===============================================
    # Save models and results
    # ===============================================
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save trained models
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.pkl"))
    nn_model.save(os.path.join(MODELS_DIR, "neural_net.keras"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # Save evaluation metrics
    results_file = os.path.join(RESULTS_DIR, "gpu_model_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    info("All GPU models trained and saved successfully!")
    info(f"Results saved to: {results_file}")
    info("Models stored in 'models/' directory")

if __name__ == "__main__":
    main()
