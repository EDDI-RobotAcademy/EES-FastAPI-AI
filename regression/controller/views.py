from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from regression.repository.regression_repository_impl import RegressionRepositoryImpl
from regression.service.regression_service_impl import RegressionServiceImpl

regression_router = APIRouter()

class RegressionRequest(BaseModel):
    user_age: int
    user_gender: int
    date_info: int

async def inject_regression_service() -> RegressionServiceImpl:
    repository = RegressionRepositoryImpl()
    return RegressionServiceImpl(repository)

@regression_router.post("/train-regression")
async def regression_train(
    regression_service: RegressionServiceImpl = Depends(inject_regression_service),
):
    try:
        regression_service.fit_model()
        return JSONResponse(content={"message": "Training completed"})
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@regression_router.post("/regression-predict")
async def regression_predict(
    requests: List[RegressionRequest],
    regression_service: RegressionServiceImpl = Depends(inject_regression_service),
):
    try:
        X_new = [[req.user_age, req.user_gender, req.date_info] for req in requests]
        predictions = regression_service.predict(X_new)
        return JSONResponse(content={"predictions": predictions.tolist()})
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )