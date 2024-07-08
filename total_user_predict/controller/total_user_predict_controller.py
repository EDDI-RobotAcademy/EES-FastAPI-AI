from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from total_user_predict.controller.request_form.total_user_predict_request_form import (
    TotalUserPredictRequestForm,
)
from total_user_predict.service.total_user_predict_service_impl import (
    TotalUserPredictServiceImpl,
)

total_user_predict_router = APIRouter()


async def inject_total_user_predict_service() -> TotalUserPredictServiceImpl:
    return TotalUserPredictServiceImpl()


@total_user_predict_router.post("/train-total-user")
async def total_user_train(
    total_user_predict_service: TotalUserPredictServiceImpl = Depends(
        inject_total_user_predict_service
    ),
):
    try:
        total_user_predict_service.train_total_user()
        return JSONResponse(content={"message": "Training completed"})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@total_user_predict_router.get("/predict-total-user")
async def total_user_predict(
    request_form: TotalUserPredictRequestForm,
    total_user_predict_service: TotalUserPredictServiceImpl = Depends(
        inject_total_user_predict_service
    ),
):
    try:
        result = total_user_predict_service.predict_total_user(
            n_days_after=request_form.n_days
        )
        return JSONResponse(content={"predicted_total_user": result})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
