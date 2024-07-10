from user_info.service.user_info_service import UserInfoService
from user_info.repository.user_info_repository_impl import UserInfoRepositoryImpl


class UserInfoServiceImpl(UserInfoService):
    def __init__(self):
        self.__user_info_repository = UserInfoRepositoryImpl()
        
    def load_user_info(self, account_id: int):
        info = self.__user_info_repository.load_user_info(account_id=account_id)
        return info