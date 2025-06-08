from fastapi import APIRouter
from pydantic import BaseModel

from services.photo_classification_service import *
photo_classification_router=APIRouter(
    prefix="/photo_classification",
    tags=["photo_classification"],
)

photo_classification_service = PhotoClassificationService()

class PathToPhoto(BaseModel):
    path: str

@photo_classification_router.post("/classify")
async def create_photo_classification(request: PathToPhoto):
    return photo_classification_service.classify_photo(request.path)
