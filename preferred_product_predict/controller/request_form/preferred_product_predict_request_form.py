from pydantic import BaseModel


class PreferredProductPredictRequestForm(BaseModel):
    gender: str
    birth_year: int
