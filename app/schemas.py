from pydantic import BaseModel

class Transaction(BaseModel):
    Transaction_Type: str
    Payment_Gateway: str
    Transaction_City: str
    Transaction_State: str
    Device_OS: str
    Merchant_Category: str
    Transaction_Channel: str
    Transaction_Status: str

    Transaction_Frequency: float
    Transaction_Amount_Deviation: float
    Days_Since_Last_Transaction: float
    amount: float
    device_id: str
    ip_address: str
    receiver: str
    message: str = ""
    
    Date: str
    Time: str