from pydantic import BaseModel


class DateInfoPredictRequestForm(BaseModel):
    n_days: int
