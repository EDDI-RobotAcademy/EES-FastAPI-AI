from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from user_withdraw_predict.controller.request_form.user_withdraw_predict_request_form import (
    UserWithdrawPredictRequestForm,
)
from user_withdraw_predict.service.user_withdraw_predict_service_impl import (
    UserWithdrawPredictServiceImpl,
)

user_withdraw_predict_router = APIRouter()


async def inject_user_withdraw_predict_service() -> UserWithdrawPredictServiceImpl:
    return UserWithdrawPredictServiceImpl()


@user_withdraw_predict_router.post("/train-user-withdraw")
async def user_withdraw_train(
    user_withdraw_predict_service: UserWithdrawPredictServiceImpl = Depends(
        inject_user_withdraw_predict_service
    ),
):
    try:
        user_withdraw_predict_service.train_user_withdraw()
        return JSONResponse(content={"message": "Training completed"})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@user_withdraw_predict_router.post("/predict-user-withdraw")
async def user_withdraw_predict(
    request_form: UserWithdrawPredictRequestForm,
    user_withdraw_predict_service: UserWithdrawPredictServiceImpl = Depends(
        inject_user_withdraw_predict_service
    ),
):
    try:
        withdraw, withdraw_prob = user_withdraw_predict_service.predict_user_withdraw(
            1 if request_form.gender == "MALE" else 0,
            request_form.birth_year,
            request_form.num_logins,
            request_form.average_login_interval,
            request_form.days_from_last_login,
            request_form.member_maintenance,
            request_form.num_orders,
            request_form.average_login_interval,
            request_form.total_spent,
            request_form.total_quantity,
            request_form.last_login_to_withdraw,
            # request_form.withdraw_reason,
        )
        return JSONResponse(
            content={
                "predicted_user_withdraw": 1 if withdraw else 0,
                "withdraw_probability": f"{withdraw_prob:.2f}",
            }
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
