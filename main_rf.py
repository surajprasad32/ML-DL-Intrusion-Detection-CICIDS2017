import os
import joblib
import json
from datetime import datetime

from src.data_preprocessing import load_cicids_data, preprocess_data
from src.model_training import train_random_forest
from src.evaluation import evaluate_model
from src.utils import info

DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"

def main():
    info("Starting the Random Forest Model for Network Intrusion Detection System")

    # Ensure directories exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load and preprocess data
    data = load_cicids_data(DATA_DIR)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Train Random Forest
    rf = train_random_forest(X_train, y_train)

    # Evaluate
    info("Evaluating Random Forest model..............")
    metrics = evaluate_model("Random Forest", rf, X_test, y_test)

    # Save model
    model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    joblib.dump(rf, model_path)
    info(f"The Model is saved at: {model_path}")

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler_rf.pkl")
    joblib.dump(scaler, scaler_path)
    info(f"Scaler is saved at: {scaler_path}")

    # Save results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"random_forest_results_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=4)
    info(f"The Metrics of the Random Forest model are saved at: {result_file}")

    info("\nThe Random Forest Training & Evaluation Completed Successfully!..........")

if __name__ == "__main__":
    main()