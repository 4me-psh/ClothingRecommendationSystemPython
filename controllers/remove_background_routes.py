from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.remove_background_service import BackgroundRemovalService

remove_bg_router=APIRouter(
    prefix="/remove_background",
    tags=["remove_background"],
)
bg_service = BackgroundRemovalService()

class RemoveBgRequest(BaseModel):
    path: str

@remove_bg_router.post("/remove")
def remove_bg_endpoint(request: RemoveBgRequest):
    try:
        data_uri = bg_service.remove_background(request.path)
        return {"image": data_uri}
    except HTTPException as e:

        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
