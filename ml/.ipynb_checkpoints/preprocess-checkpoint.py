import pandas as pd
import numpy as np
import joblib

# ---------------- LOAD ---------------- #
def load_data(path):
    return pd.read_csv(path)


# ---------------- CLEAN ---------------- #
def clean_data(df):
    # Drop ID if exists
    df = df.drop(columns=["id"], errors="ignore")
    return df
    
# ---------------- FEATURE ENGINEERING ---------------- #
def feature_engineering(df):

    # Log transform (important for skewed amount)
    df["amount_log"] = np.log1p(df["amount"])

    # High amount flag
    df["is_high_amount"] = (df["amount"] > 2000).astype(int)

    # Normalize Amount (optional but useful)
    df["amount_zscore"] = (
        (df["amount"] - df["amount"].mean()) /
        (df["amount"].std() + 1e-5)
    )

    return df

def handle_missing(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df

# ---------------- DROP ---------------- #
def drop_irrelevant(df):
    drop_cols = [
        "Transaction_ID",
        "Customer_ID",
        "Merchant_ID",
        "Device_ID",
        "IP_Address",
        "Date",
        "Time"
    ]
    return df.drop(columns=drop_cols, errors="ignore")
# ---------------- SPLIT ---------------- #
def prepare_data(df):
    X = df.drop("fraud", axis=1)
    y = df["fraud"]
    return X, y

def encode_data(df):
    categorical_cols = [
        "Transaction_Type",
        "Transaction_City",
        "Transaction_State",
        "Device_OS",
        "Transaction_Channel"
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.select_dtypes(include=[np.number])

    return df
# ---------------- LOAD COLUMNS ---------------- #
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "../ml/columns.pkl"))


# ---------------- PREDICT PIPELINE ---------------- #
def prepare_input(data):
    df = pd.DataFrame([data])

    df = feature_engineering(df)
    df = drop_irrelevant(df)
    df = encode_data(df)
    df = handle_missing(df)

    columns = load_columns()
    df = df.reindex(columns=columns, fill_value=0)

    return df