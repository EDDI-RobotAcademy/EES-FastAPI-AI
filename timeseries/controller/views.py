from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from timeseries.repository.timeseries_repository_impl import TimeseriesRepositoryImpl
from timeseries.service.timeseries_service_impl import TimeseriesServiceImpl

timeseries_router = APIRouter()

async def inject_timeseries_service() -> TimeseriesServiceImpl:
    repository = TimeseriesRepositoryImpl()
    return TimeseriesServiceImpl(repository)

@timeseries_router.post("/train-timeseries")
async def timeseries_train(
    timeseries_service: TimeseriesServiceImpl = Depends(inject_timeseries_service),
):
    try:
        timeseries_service.fit_model()
        return JSONResponse(content={"message": "Training completed"})
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
@timeseries_router.get("/timeseries-forecast")
async def timeseries_forecast(
    steps: int,
    timeseries_service: TimeseriesServiceImpl = Depends(inject_timeseries_service),
):
    try:
        forecast = timeseries_service.forecast(steps)
        return JSONResponse(content={"forecast": forecast.tolist()})
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )