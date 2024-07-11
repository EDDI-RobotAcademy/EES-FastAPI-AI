from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from date_info_predict.controller.request_form.date_info_predict_request_form import (
    DateInfoPredictRequestForm,
)
from date_info_predict.service.date_info_predict_service_impl import (
    DateInfoPredictServiceImpl,
)

date_info_predict_router = APIRouter()


async def inject_date_info_predict_service() -> DateInfoPredictServiceImpl:
    return DateInfoPredictServiceImpl()


@date_info_predict_router.post("/train-date-info")
async def date_info_train(
    date_info_predict_service: DateInfoPredictServiceImpl = Depends(
        inject_date_info_predict_service
    ),
):
    try:
        date_info_predict_service.train_date_info()
        return JSONResponse(content={"message": "Training completed"})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@date_info_predict_router.post("/predict-total-user")
async def total_user_predict(
    request_form: DateInfoPredictRequestForm,
    date_info_predict_service: DateInfoPredictServiceImpl = Depends(
        inject_date_info_predict_service
    ),
):
    try:
        result = date_info_predict_service.predict_date_info(
            n_days_after=request_form.n_days, feature="total_user"
        )
        return JSONResponse(content={"predicted_total_user": result})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@date_info_predict_router.post("/predict-profit")
async def total_profit_predict(
    request_form: DateInfoPredictRequestForm,
    date_info_predict_service: DateInfoPredictServiceImpl = Depends(
        inject_date_info_predict_service
    ),
):
    try:
        result = date_info_predict_service.predict_date_info(
            n_days_after=request_form.n_days, feature="profit"
        )
        return JSONResponse(content={"predicted_profit": result})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
