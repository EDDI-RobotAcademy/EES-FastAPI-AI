from abc import ABC, abstractmethod


class UserInfoService(ABC):
    @abstractmethod
    def load_user_info(self, account_id: int):
        pass