# 💳 Credit Scoring & Risk Assessment System

This project builds a Machine Learning system to predict customer creditworthiness using the German Credit Dataset.

## 🚀 Features
- Data Cleaning (Null & Duplicate Removal)
- Feature Engineering
- SMOTE for Class Imbalance
- Random Forest Model
- ROC-AUC Evaluation
- Streamlit Web App
- Adjustable Business Approval Threshold

## 📊 Model Performance
- Accuracy: ~80–85%
- ROC-AUC: ~0.82+
- Balanced classification using SMOTE

## 🛠 Tech Stack
- Python
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- Pandas & NumPy

## ▶ How to Run

pip install -r requirements.txt
python src/train.py
streamlit run app.py