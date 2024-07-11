from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from preferred_product_predict.controller.request_form.preferred_product_predict_request_form import (
    PreferredProductPredictRequestForm,
)
from preferred_product_predict.service.preferred_product_predict_service_impl import (
    PreferredProductPredictServiceImpl,
)

preferred_product_predict_router = APIRouter()


async def inject_preferred_product_predict_service() -> PreferredProductPredictServiceImpl:
    return PreferredProductPredictServiceImpl()


@preferred_product_predict_router.post("/train-preferred-product")
async def preferred_product_train(
    preferred_product_predict_service: PreferredProductPredictServiceImpl = Depends(
        inject_preferred_product_predict_service
    ),
):
    try:
        preferred_product_predict_service.train_preferred_product()
        return JSONResponse(content={"message": "Training completed"})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@preferred_product_predict_router.post("/predict-preferred-product")
async def preferred_product_predict(
    request_form: PreferredProductPredictRequestForm,
    preferred_product_predict_service: PreferredProductPredictServiceImpl = Depends(
        inject_preferred_product_predict_service
    ),
):
    try:
        preferred_product, prefer_prob = preferred_product_predict_service.predict_preferred_product(
            1 if request_form.gender == "MALE" else 0,
            request_form.birth_year,
        )
        return JSONResponse(
            content={
                "predicted_preferred_product": preferred_product,
                "prefer_probability": f"{prefer_prob:.2f}",
            }
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
