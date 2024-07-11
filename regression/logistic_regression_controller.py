from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware
import os



regression_router = APIRouter()

@regression_router.get("/logistic-regression")
def logistic_regression_test():
    # 파일 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("★★★기본경로 : ", base_dir)
    file_path = os.path.join(base_dir, 'assets', 'dataset', 'orders.csv')

    # CSV 파일 로드
    data = pd.read_csv(file_path)

    # 필요한 특성과 레이블 선택
    # 예를 들어 성별(gender)과 출생 연도(birth_year)를 사용하여 제품 구매 여부를 예측한다고 가정
    data['label'] = data['product_id'].apply(lambda x: 1 if x == 1 else 0)  # 예시로 특정 제품(product_id == 1) 구매 여부
    X = data[['birth_year', 'quantity']]  # 특성 컬럼 이름을 실제 데이터에 맞게 변경해야 합니다.
    y = data['label']  # 레이블 컬럼 이름을 실제 데이터에 맞게 변경해야 합니다.

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 로지스틱 회귀 모델 학습
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 모델 예측
    y_pred = model.predict(X_test)

    # 모델 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    coef = model.coef_
    intercept = model.intercept_

    # 결정 경계 계산
    x_values = np.linspace(X['birth_year'].min(), X['birth_year'].max(), 100)
    y_values = -(coef[0][0] * x_values + intercept[0]) / coef[0][1]

    return JSONResponse(content={
        "accuracy": accuracy,
        "coefficients": coef.tolist(),
        "intercept": intercept.tolist(),
        "data_point": {
            "X": X.values.tolist(),
            "y": y.tolist()
        },
        "decision_boundary": {
            "x_values": x_values.tolist(),
            "y_values": y_values.tolist()
        }
    })


