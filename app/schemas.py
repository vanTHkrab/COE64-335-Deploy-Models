from pydantic import BaseModel

class PredictRequest(BaseModel):
    PROV_ID: int
    month_sin: float
    month_cos: float
