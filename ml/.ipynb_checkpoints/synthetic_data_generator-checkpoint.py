import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ---------------- SETTINGS ---------------- #
NUM_USERS = 2000
TXNS_PER_USER = 30

cities = ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad"]
states = ["MH", "DL", "KA", "WB", "TS"]
devices = ["Android", "iOS", "Windows"]
txn_types = ["Payment", "Transfer", "Bill Payment", "Recharge"]
channels = ["Mobile", "Web", "POS"]

# ---------------- USER PROFILES ---------------- #
users = []

for user_id in range(NUM_USERS):
    base_amount = np.random.randint(500, 5000)
    freq = np.random.randint(2, 10)
    home_city = random.choice(cities)

    users.append({
        "user_id": user_id,
        "base_amount": base_amount,
        "freq": freq,
        "city": home_city
    })

# ---------------- GENERATE TRANSACTIONS ---------------- #
data = []

start_date = datetime(2024, 1, 1)

for user in users:
    last_txn_time = start_date

    for i in range(TXNS_PER_USER):

        # normal behavior
        amount = np.random.normal(user["base_amount"], user["base_amount"] * 0.3)
        amount = max(50, amount)

        txn_time = last_txn_time + timedelta(
            hours=np.random.randint(1, 72)
        )

        hour = txn_time.hour
        deviation = abs(amount - user["base_amount"]) / (user["base_amount"] + 1)

        fraud = 0

        # ---------------- FRAUD SCENARIOS ---------------- #

        # 1️⃣ High amount at night
        if amount > 20000 and hour < 6:
            fraud = 1

        # 2️⃣ Velocity attack
        if np.random.rand() < 0.05:
            amount = np.random.randint(10000, 50000)
            deviation = 1
            fraud = 1

        # 3️⃣ Account takeover (city change + high amount)
        txn_city = user["city"]
        if np.random.rand() < 0.03:
            txn_city = random.choice(cities)
            if txn_city != user["city"]:
                amount = np.random.randint(15000, 60000)
                fraud = 1

        # 4️⃣ Sudden spike
        if deviation > 0.8 and np.random.rand() < 0.5:
            fraud = 1

        # ---------------- RECORD ---------------- #
        data.append({
            "Transaction_ID": f"T{user['user_id']}_{i}",
            "Date": txn_time.strftime("%Y-%m-%d"),
            "Time": txn_time.strftime("%H:%M:%S"),
            "Customer_ID": f"C{user['user_id']}",
            "Transaction_City": txn_city,
            "Transaction_State": random.choice(states),
            "Device_OS": random.choice(devices),
            "Transaction_Type": random.choice(txn_types),
            "Transaction_Channel": random.choice(channels),
            "Transaction_Frequency": user["freq"],
            "Transaction_Amount_Deviation": deviation,
            "Days_Since_Last_Transaction": np.random.randint(0, 10),
            "amount": round(amount, 2),
            "fraud": fraud
        })

        last_txn_time = txn_time

# ---------------- SAVE ---------------- #
df = pd.DataFrame(data)

print("\n📊 Dataset Shape:", df.shape)
print("\n📊 Fraud Distribution:")
print(df["fraud"].value_counts())

df.to_csv("D:/paysafe-backend/Data/Raw/synthetic_fraud.csv", index=False)

print("\n✅ Synthetic dataset created!")