from pydantic import BaseModel


class TotalUserPredictRequestForm(BaseModel):
    n_days: int
