from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from user_spent_predict.controller.request_form.user_spent_predict_request_form import (
    UserSpentPredictRequestForm,
)
from user_spent_predict.service.user_spent_predict_service_impl import (
    UserSpentPredictServiceImpl,
)

user_spent_predict_router = APIRouter()


async def inject_user_spent_predict_service() -> UserSpentPredictServiceImpl:
    return UserSpentPredictServiceImpl()


@user_spent_predict_router.post("/train-user-spent")
async def user_spent_train(
    user_spent_predict_service: UserSpentPredictServiceImpl = Depends(
        inject_user_spent_predict_service
    ),
):
    try:
        user_spent_predict_service.train_user_spent()
        return JSONResponse(content={"message": "Training completed"})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@user_spent_predict_router.post("/predict-user-spent")
async def user_spent_predict(
    request_form: UserSpentPredictRequestForm,
    user_spent_predict_service: UserSpentPredictServiceImpl = Depends(
        inject_user_spent_predict_service
    ),
):
    try:
        expected_spent = user_spent_predict_service.predict_user_spent(
            1 if request_form.gender == "MALE" else 0,
            request_form.birth_year,
            request_form.num_logins,
            request_form.average_login_interval,
            request_form.days_from_last_login,
            request_form.member_maintenance,
            request_form.num_orders,
            request_form.average_login_interval,
        )
        return JSONResponse(
            content={
                "expected_spent": f"{expected_spent}",
            }
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
