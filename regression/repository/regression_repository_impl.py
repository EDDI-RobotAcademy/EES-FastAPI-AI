import os
import pandas as pd

from regression.repository.regression_repository import RegressionRepository


class RegressionRepositoryImpl(RegressionRepository):
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.date_info_path = os.path.join(base_dir, '../../assets/dataset/date_info.csv')
        self.orders_path = os.path.join(base_dir, '../../assets/dataset/orders.csv')
        self.user_info_path = os.path.join(base_dir, '../../assets/dataset/user_info.csv')

    def load_data(self):
        try:
            # 데이터 로드
            date_info = pd.read_csv(self.date_info_path)
            orders = pd.read_csv(self.orders_path)
            user_info = pd.read_csv(self.user_info_path)

            # 데이터 전처리
            date_info['date'] = pd.to_datetime(date_info['date'])
            orders['ordered_at'] = pd.to_datetime(orders['ordered_at'])
            orders.rename(columns={'ordered_at': 'order_date'}, inplace=True)
            user_info = user_info.ffill()

            # 데이터 병합
            merged_data = pd.merge(orders, user_info, on=['gender', 'birth_year'], how='left')
            merged_data = pd.merge(merged_data, date_info, left_on='order_date', right_on='date', how='left')

            # 로그 출력
            print("Merged Data Columns: ", merged_data.columns)

            if 'order_date' not in merged_data.columns:
                raise ValueError("order_date column not found in merged data")

            merged_data.drop(columns=['date'], inplace=True)
            return merged_data
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def get_order_data(self):
        try:
            data = self.load_data()
            return data[['total_price', 'gender', 'birth_year', 'order_date']]
        except Exception as e:
            print(f"Error getting order data: {e}")
            raise
