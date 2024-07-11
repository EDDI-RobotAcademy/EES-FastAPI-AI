import numpy as np
import pandas as pd

from user_info.repository.user_info_repository import UserInfoRepository


class UserInfoRepositoryImpl(UserInfoRepository):        
    def load_user_info(self, account_id: int):
        account_id = account_id - 1
        df = pd.read_csv('assets/dataset/user_info.csv')
        df = df[df['withdraw'] == False]
        df = df.drop(columns=['withdraw', 'withdraw_reason', 'last_login_to_withdraw'])
        keys = df.keys().to_list()
        df = df[df.index == account_id]
        df = df.iloc[0]
        if len(df) == 0:
            raise ValueError('No user found')
        df = df.to_list()
        data = [x.item() if isinstance(x, np.generic) else x for x in df]
        info = {keys[i]: data[i] for i in range(len(keys))}
        return info