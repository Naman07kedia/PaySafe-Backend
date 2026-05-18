import joblib
from sklearn.ensemble import IsolationForest
import pandas as pd

from preprocess import (
    load_data,
    clean_data,
    feature_engineering,
    encode_data,
    drop_irrelevant,
    prepare_data
)

# Load dataset
df = load_data("data/raw/upi_fraud.csv")

df = clean_data(df)
df = feature_engineering(df)
df = drop_irrelevant(df)
df = encode_data(df)

X, _ = prepare_data(df)

# Train anomaly model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# Save
joblib.dump(model, "ml/anomaly.pkl")

print("Anomaly model saved!")