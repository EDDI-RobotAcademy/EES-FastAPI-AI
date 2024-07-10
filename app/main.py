import os
import sys

from regression.controller.views import regression_router
from timeseries.controller.views import timeseries_router

sys.path.append('..')

from date_info_predict.controller.date_info_predict_controller import date_info_predict_router
from user_withdraw_predict.controller.user_withdraw_predict_controller import user_withdraw_predict_router
from user_spent_predict.controller.user_spent_predict_controller import user_spent_predict_router

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.include_router(date_info_predict_router)
app.include_router(user_withdraw_predict_router)
app.include_router(user_spent_predict_router)
app.include_router(timeseries_router)
app.include_router(regression_router)

allow_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

