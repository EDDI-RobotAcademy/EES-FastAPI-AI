import sys
sys.path.append('..')

from total_user_predict.controller.total_user_predict_controller import total_user_predict_router
from user_withdraw_predict.controller.user_withdraw_predict_controller import user_withdraw_predict_router

from fastapi import FastAPI


app = FastAPI()

app.include_router(total_user_predict_router)
app.include_router(user_withdraw_predict_router)