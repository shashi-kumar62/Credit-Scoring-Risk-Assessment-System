import joblib
import numpy as np
from sklearn.metrics import classification_report
from preprocess import load_and_clean_data
from sklearn.model_selection import train_test_split

df = load_and_clean_data("data/german_credit_data.csv")

X = df.drop("kredit", axis=1)
y = df["kredit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = joblib.load("../models/final_model.pkl")
y_prob = model.predict_proba(X_test)[:,1]

for threshold in [0.3, 0.4, 0.5, 0.6]:
    print(f"\nThreshold: {threshold}")
    y_pred_custom = (y_prob >= threshold).astype(int)
    print(classification_report(y_test, y_pred_custom))