from sqlalchemy import Column, Integer, Float, String
from app.database import Base

class TransactionDB(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)

    amount = Column(Float)
    fraud = Column(Integer)
    risk_score = Column(Float)

    anomaly = Column(Integer)
    anomaly_score = Column(Float)

    city = Column(String)
    state = Column(String)
    transaction_type = Column(String)
    
from sqlalchemy import Column, Integer, String

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    password = Column(String)
    role = Column(String)   # admin / user
    avg_amount = Column(Float, default=0)
    last_transaction_time = Column(String, nullable=True)
    last_city = Column(String, nullable=True)
    txn_count_24h = Column(Integer, default=0)
    last_device_id = Column(String, nullable=True)
    last_ip = Column(String, nullable=True)