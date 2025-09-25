from pydantic import BaseModel

class PredictRequest(BaseModel):
    PROV_ID: int
    month_id: int
