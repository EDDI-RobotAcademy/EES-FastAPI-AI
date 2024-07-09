import sys
sys.path.append('..')

from date_info_predict.controller.date_info_predict_controller import date_info_predict_router
from user_withdraw_predict.controller.user_withdraw_predict_controller import user_withdraw_predict_router

from fastapi import FastAPI


app = FastAPI()

app.include_router(date_info_predict_router)
app.include_router(user_withdraw_predict_router)