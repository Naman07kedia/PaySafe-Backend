import joblib
import pandas as pd
import numpy as np
import shap
import re
from ml.preprocess import prepare_input

# ---------------- LOAD MODELS ---------------- #
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "../ml/model.pkl")
)

columns = joblib.load(
    os.path.join(BASE_DIR, "../ml/columns.pkl")
)

# Optional anomaly model
try:
    anomaly_model = joblib.load("ml/anomaly.pkl")
except:
    anomaly_model = None

# SHAP
try:
    explainer = shap.TreeExplainer(model)
except:
    explainer = None

# ---------------- NLP SCAM DETECTION ---------------- #
scam_patterns = [
    r"otp", r"urgent", r"verify", r"kyc",
    r"blocked", r"refund", r"click.*link",
    r"win.*prize", r"limited.*time", r"account.*suspend"
]

def analyze_message(message: str):
    if not message:
        return 0.0, []

    message = message.lower()
    score = 0
    triggers = []

    for pattern in scam_patterns:
        if re.search(pattern, message):
            score += 0.1
            triggers.append(pattern)

    if "http" in message or "www" in message:
        score += 0.2
        triggers.append("suspicious_link")

    if any(word in message for word in ["urgent", "now", "immediately"]):
        score += 0.1
        triggers.append("pressure_language")

    if any(word in message for word in ["bank", "account", "otp"]):
        score += 0.1
        triggers.append("sensitive_request")

    return min(score, 1.0), triggers


# ---------------- RULE ENGINE ---------------- #
def rule_based_score(data):
    score = 0

    if data.get("amount", 0) > 50000:
        score += 0.3

    if data.get("Transaction_Amount_Deviation", 0) > 0.7:
        score += 0.3

    if data.get("Transaction_Frequency", 0) > 15:
        score += 0.2

    if data.get("Days_Since_Last_Transaction", 0) > 10:
        score += 0.1

    try:
        hour = int(data.get("Time", "12:00:00").split(":")[0])
        if hour < 6:
            score += 0.1
    except:
        pass

    return min(score, 1.0)


# ---------------- MAIN DETECTOR ---------------- #
def detect_fraud(data: dict):

    # ---------------- ML ---------------- #
    X = prepare_input(data)
    X = X.select_dtypes(include=[np.number])

    ml_score = model.predict_proba(X)[0][1]

    # ---------------- RULE ---------------- #
    rule_score = rule_based_score(data)

    # ---------------- BEHAVIOR ---------------- #
    behavior_score = 0
    user_avg = data.get("user_avg", 10000)

    if data.get("amount", 0) > 3 * user_avg:
        behavior_score += 0.4

    if data.get("Transaction_City") != data.get("last_city"):
        behavior_score += 0.2

    # ---------------- DEVICE ---------------- #
    device_score = 0

    if data.get("Device_ID") != data.get("last_device_id"):
        device_score += 0.3

    if data.get("IP_Address") != data.get("last_ip"):
        device_score += 0.2

    # ---------------- NETWORK ---------------- #
    network_score = 0
    if data.get("receiver") in ["fraud123", "scammer999"]:
        network_score += 0.6

    # ---------------- NLP ---------------- #
    nlp_score, triggers = analyze_message(data.get("message", ""))

    # ---------------- ANOMALY ---------------- #
    anomaly_score = 0
    anomaly_flag = 0

    if anomaly_model:
        try:
            anomaly_score = anomaly_model.decision_function(X)[0]
            anomaly_flag = anomaly_model.predict(X)[0]
        except:
            pass

    # ---------------- FINAL RISK ---------------- #
    final_risk = (
        ml_score * 0.35 +
        rule_score * 0.20 +
        behavior_score * 0.20 +
        device_score * 0.10 +
        network_score * 0.10 +
        nlp_score * 0.05
    )

    if nlp_score > 0.5 and rule_score > 0.2:
        final_risk += 0.2

    if network_score > 0.5:
        final_risk += 0.3

    final_risk = min(final_risk, 1.0)

    # ---------------- LABEL ---------------- #
    if final_risk < 0.3:
        label = "SAFE"
    elif final_risk < 0.7:
        label = "SUSPICIOUS"
    else:
        label = "HIGH RISK"

    # ---------------- SHAP ---------------- #
top_features = []

if explainer:
    try:
        shap_values = explainer.shap_values(X)

        shap_df = pd.DataFrame({
            "feature": X.columns,
            "impact": shap_values[0]
        })

        shap_df["abs"] = shap_df["impact"].abs()

        top_features_df = shap_df.sort_values(
            by="abs",
            ascending=False
        ).head(5)

        top_features = [
            {
                "feature": row["feature"],
                "impact": float(row["impact"])
            }
            for _, row in top_features_df.iterrows()
        ]

    except Exception as e:
        print("SHAP ERROR:", e)

    # ---------------- HUMAN REASONS ---------------- #
    reasons = []

    if data.get("amount", 0) > 50000:
        reasons.append("💸 High transaction amount")

    if data.get("Transaction_Amount_Deviation", 0) > 0.7:
        reasons.append("📊 Unusual spending behavior")

    if data.get("Transaction_Frequency", 0) > 15:
        reasons.append("⚡ Too many transactions in short time")

    if device_score > 0:
        reasons.append("📱 New or unrecognized device")

    if network_score > 0:
        reasons.append("🚫 Risky receiver detected")

    if nlp_score > 0:
        reasons.append("🧠 Suspicious message content")

    if anomaly_flag == -1:
        reasons.append("⚠️ Anomalous behavior detected")

    if not reasons:
        reasons.append("No major risk signals detected")

    # ---------------- RETURN ---------------- #
    return {
        "fraud": int(final_risk > 0.7),
        "risk_score": float(final_risk),
        "risk_level": label,
        "ml_score": float(ml_score),
        "rule_score": float(rule_score),
        "behavior_score": float(behavior_score),
        "device_score": float(device_score),
        "network_score": float(network_score),
        "nlp_score": float(nlp_score),
        "anomaly": int(anomaly_flag),
        "anomaly_score": float(anomaly_score),
        "reasons": reasons,
        "top_features": top_features
    }