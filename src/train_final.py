import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/german_credit_data.csv")

# Remove duplicates and nulls
df = df.drop_duplicates()
df = df.dropna()

# Convert target
df["kredit"] = df["kredit"].replace({1: 1, 2: 0})

# Feature Engineering
df["credit_per_month"] = df["hoehe"] / df["laufzeit"]

X = df.drop("kredit", axis=1)
y = df["kredit"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Pipeline (SMOTE + Model)
# -----------------------------
pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/final_pipeline.pkl")

print("\nModel saved successfully.")