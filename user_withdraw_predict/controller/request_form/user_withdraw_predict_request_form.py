from pydantic import BaseModel


class UserWithdrawPredictRequestForm(BaseModel):
    gender: str
    birth_year: int
    num_logins: int
    average_login_interval: float
    days_from_last_login: int
    member_maintenance: int
    num_orders: int
    average_order_interval: float
    total_spent: int
    total_quantity: int
    last_login_to_withdraw: int = 0
