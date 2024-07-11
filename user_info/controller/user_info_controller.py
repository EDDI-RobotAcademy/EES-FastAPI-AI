from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from user_info.service.user_info_service_impl import UserInfoServiceImpl
from user_info.controller.request_form.user_info_request_form import UserInfoRequestForm


user_info_router = APIRouter()


async def inject_user_info_service() -> UserInfoServiceImpl:
    return UserInfoServiceImpl()


@user_info_router.post("/user-info")
async def user_info(
    request_form: UserInfoRequestForm,
    user_info_service: UserInfoServiceImpl = Depends(inject_user_info_service),
):
    try:
        info = user_info_service.load_user_info(account_id=request_form.account_id)
        return JSONResponse(content=info)

    except Exception as e:
        return JSONResponse(
            content={"error": str("이미 탈퇴한 고객입니다.")}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
