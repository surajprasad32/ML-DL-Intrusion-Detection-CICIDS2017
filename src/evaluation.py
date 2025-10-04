from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from src.utils import info
import numpy as np

def evaluate_model(name, clf, X_test, y_test, is_keras=False):
    info(f"Evaluating {name}...")
    if is_keras:
        y_pred_proba = clf.predict(X_test).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
    return {"acc": acc, "f1": f1, "auc": auc}
