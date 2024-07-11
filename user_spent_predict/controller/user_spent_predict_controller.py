from fastapi import APIRouter, Depends, Query, FastAPI, status
from fastapi.responses import JSONResponse

from user_spent_predict.service.user_spent_predict_service_impl import (
    UserSpentPredictServiceImpl,
)

user_spent_predict_router = APIRouter()

async def inject_user_spent_predict_service() -> UserSpentPredictServiceImpl:
    return UserSpentPredictServiceImpl()

@user_spent_predict_router.post("/train-user-spent")
async def user_spent_train(
    user_spent_predict_service: UserSpentPredictServiceImpl = Depends(inject_user_spent_predict_service),
):
    try:
        user_spent_predict_service.train_user_spent()
        return JSONResponse(content={"message": "Training completed"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@user_spent_predict_router.post("/predict-user-spent")
async def user_spent_predict(
    gender: str = Query(..., alias="gender"),
    birth_year: int = Query(..., alias="birth_year"),
    num_logins: int = Query(..., alias="num_logins"),
    average_login_interval: int = Query(..., alias="average_login_interval"),
    days_from_last_login: int = Query(..., alias="days_from_last_login"),
    member_maintenance: int = Query(..., alias="member_maintenance"),
    num_orders: int = Query(..., alias="num_orders"),
    average_order_interval: int = Query(..., alias="average_order_interval"),
    user_spent_predict_service: UserSpentPredictServiceImpl = Depends(inject_user_spent_predict_service),
):
    try:
        expected_spent = user_spent_predict_service.predict_user_spent(
            1 if gender == "MALE" else 0,
            birth_year,
            num_logins,
            average_login_interval,
            days_from_last_login,
            member_maintenance,
            num_orders,
            average_order_interval,
        )
        return JSONResponse(content={"expected_spent": f"{expected_spent}"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


