from fastapi import FastAPI, Depends, Header, HTTPException
from sqlalchemy.orm import Session
from jose import jwt

from app.schemas import Transaction
from app.detector import detect_fraud
from app.database import engine, Base, SessionLocal
from app.models import TransactionDB, User
from app.auth import hash_password, verify_password, create_token

from fastapi.middleware.cors import CORSMiddleware

# ---------------- APP ---------------- #
app = FastAPI(title="PaySafe Fraud Detection API")

# ---------------- CORS ---------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATABASE ---------------- #
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- AUTH ---------------- #
def get_current_user(token: str = Header(...)):
    try:
        payload = jwt.decode(token, "supersecret", algorithms=["HS256"])
        return payload["sub"]
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------------- ROUTES ---------------- #

@app.get("/")
def home():
    return {"message": "PaySafe API Running"}

# ---------------- REGISTER ---------------- #
@app.post("/register")
def register(username: str, password: str, db: Session = Depends(get_db)):

    if db.query(User).filter(User.username == username).first():
        return {"error": "User already exists"}

    user = User(
        username=username,
        password=hash_password(password),
        role="user",
        avg_amount=0,
        txn_count_24h=0
    )

    db.add(user)
    db.commit()

    return {"message": "User created successfully"}

# ---------------- LOGIN ---------------- #
@app.post("/login")
def login(username: str, password: str, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(password, user.password):
        return {"error": "Invalid credentials"}

    token = create_token({"sub": user.username})

    return {"access_token": token}

# ---------------- GET TRANSACTIONS ---------------- #
@app.get("/transactions")
def get_transactions(
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user)
):
    return db.query(TransactionDB).all()

# ---------------- PREDICT ---------------- #
@app.post("/predict")
def predict(
    txn: Transaction,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user)
):

    # ---------------- GET USER ---------------- #
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # ---------------- PREPARE DATA ---------------- #
    txn_data = txn.dict()

    txn_data.update({
        "user_avg": user.avg_amount or 10000,
        "last_city": user.last_city,
        "txn_count_24h": user.txn_count_24h or 0,
        "last_device_id": user.last_device_id,
        "last_ip": user.last_ip
    })

    # ---------------- FRAUD DETECTION ---------------- #
    result = detect_fraud(txn_data)

    anomaly = result.get("anomaly", 0)
    anomaly_score = result.get("anomaly_score", 0.0)

    # ---------------- ALERT ---------------- #
    if result["fraud"] == 1 or anomaly == -1:
        print("🚨 ALERT: Suspicious transaction detected!")

    # ---------------- SAVE TRANSACTION ---------------- #
    new_txn = TransactionDB(
        amount=txn.amount,
        fraud=result["fraud"],
        risk_score=result["risk_score"],
        anomaly=anomaly,
        anomaly_score=anomaly_score,
        city=txn.Transaction_City,
        state=txn.Transaction_State,
        transaction_type=txn.Transaction_Type
    )

    db.add(new_txn)

    # ---------------- UPDATE USER PROFILE ---------------- #
    user.avg_amount = (
        (user.avg_amount + txn.amount) / 2
        if user.avg_amount else txn.amount
    )

    user.last_city = txn.Transaction_City
    user.txn_count_24h = (user.txn_count_24h or 0) + 1

    # Match schema fields properly
    user.last_device_id = getattr(txn, "device_id", None)
    user.last_ip = getattr(txn, "ip_address", None)

    db.commit()

    return result