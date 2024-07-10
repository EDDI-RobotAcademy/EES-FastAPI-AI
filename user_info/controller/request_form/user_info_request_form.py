from pydantic import BaseModel


class UserInfoRequestForm(BaseModel):
    account_id: int