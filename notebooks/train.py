
import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Load processed data
data = joblib.load("outputs/processed_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=42, class_weight="balanced"
    )
}

results = []

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

best_model = None
best_score = -1
best_name = ""

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    print(f"\n{name}")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Select using F1 score
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name

# Save comparison
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/model_comparison.csv", index=False)

# Save best model
joblib.dump(best_model, "models/loan_model.pkl")

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print("\nBest Model:", best_name)
print("Best F1 Score:", round(best_score, 4))
print("Model saved: models/loan_model.pkl")

