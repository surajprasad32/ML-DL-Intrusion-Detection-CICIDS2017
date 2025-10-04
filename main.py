import os
import joblib
from src.data_preprocessing import load_cicids_data, preprocess_data
from src.model_training import train_random_forest, train_xgboost, train_keras_nn
from src.evaluation import evaluate_model
from src.utils import info

DATA_DIR = "data"  # Folder where your CSVs will go

def main():
    info("Starting Network Intrusion Detection System (CICIDS2017)")

    data = load_cicids_data(DATA_DIR)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    rf = train_random_forest(X_train, y_train)
    xg = train_xgboost(X_train, y_train)
    nn = train_keras_nn(X_train, y_train, X_train.shape[1])

    results = {}
    results["RandomForest"] = evaluate_model("Random Forest", rf, X_test, y_test)
    results["XGBoost"] = evaluate_model("XGBoost", xg, X_test, y_test)
    results["NeuralNet"] = evaluate_model("Neural Network", nn, X_test, y_test, is_keras=True)

    info("\nSummary of all model results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(xg, "models/xgboost.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    nn.save("models/neural_net.h5")

    info("Models saved in 'models/' directory")

if __name__ == "__main__":
    main()