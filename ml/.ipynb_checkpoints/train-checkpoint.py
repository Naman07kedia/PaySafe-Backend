import numpy as np
import pandas as pd
import joblib
import json

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from preprocess import (
    load_data,
    clean_data,
    drop_irrelevant,
    handle_missing,
    encode_data,
    feature_engineering,
    prepare_data
)

# ---------------- LOAD ---------------- #
df = load_data("D:/paysafe-backend/Data/Raw/synthetic_fraud.csv")
df.columns = df.columns.str.strip()   # remove spaces
print(df.columns.tolist())
print("\n📊 Class Distribution:")
print(df["fraud"].value_counts())

# ---------------- PROCESS ---------------- #
df = clean_data(df)
df = feature_engineering(df)
df = drop_irrelevant(df)
df = encode_data(df)
df = handle_missing(df)

X, y = prepare_data(df)

# ---------------- SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- HANDLE IMBALANCE ---------------- #
fraud = (y_train == 1).sum()
normal = (y_train == 0).sum()

scale_pos_weight = normal / fraud

print("Scale Pos Weight:", scale_pos_weight)

# ---------------- MODEL ---------------- #
model = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="auc"
)
# 🚨 Ensure only numeric features
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test[X_train.columns]

# ---------------- TRAIN ---------------- #
model.fit(X_train, y_train)

feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n🔝 Top Features:")
print(feat_imp.head(15))
# ---------------- PREDICT ---------------- #
y_prob = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ---------------- THRESHOLD TUNING ---------------- #
best_f1 = 0
best_threshold = 0

for t in np.arange(0.2, 0.8, 0.02):
    y_pred_temp = (y_prob > t).astype(int)
    f1 = f1_score(y_test, y_pred_temp)

    print(f"Threshold {round(t,2)} → F1: {round(f1,4)}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\n✅ Best Threshold:", best_threshold)
print("✅ Best F1:", best_f1)

# FINAL PREDICTION
y_pred = (y_prob > best_threshold).astype(int)

# ---------------- METRICS ---------------- #
print("\n📊 MODEL PERFORMANCE")
print("-" * 40)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- SAVE ---------------- #
joblib.dump(model, "ml/model.pkl")
joblib.dump(X_train.columns.tolist(), "ml/columns.pkl")

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "best_threshold": float(best_threshold)
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f)

print("\n✅ Model + columns saved!")