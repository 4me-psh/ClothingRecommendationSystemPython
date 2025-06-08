from typing import List, Union, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.image_generation_service import ImageGenerationService

class PersonInfo(BaseModel):
    gender: Optional[str] = None
    skinTone: Optional[str] = None
    hairColor: Optional[str] = None
    height: Optional[float] = None
    age: Optional[int] = None

class GenerateRequest(BaseModel):
    image_paths: List[str] = Field(..., alias="image_paths")
    person: Union[str, PersonInfo]

class PieceOfClothesInfo(BaseModel):
    name: Optional[str]
    color: Optional[str]
    material: Optional[str]
    characteristics: Optional[List[str]] = []

class PersonDescription(BaseModel):
    gender: Optional[str]
    skinTone: Optional[str]
    hairColor: Optional[str]
    height: Optional[float]
    age: Optional[int]

class GenerateStableRequest(BaseModel):
    person: PersonDescription
    clothes: List[PieceOfClothesInfo]

image_generation_router = APIRouter(
    prefix="/image_generation",
    tags=["image_generation"],
)

image_generation_service = ImageGenerationService()

@image_generation_router.post("/generate")
async def create_image_generation(payload: GenerateRequest):
    if not isinstance(payload.person, str):
        person_payload = {
            "gender": getattr(payload.person, "gender", None),
            "skinTone": getattr(payload.person, "skinTone", None),
            "hairColor": getattr(payload.person, "hairColor", None),
            "height": getattr(payload.person, "height", None),
            "age": getattr(payload.person, "age", None)
        }
    else:
        person_payload = payload.person

    return {
        "base64_image": image_generation_service.generate_image(
            image_paths=payload.image_paths,
            person_payload=person_payload
        )
    }


@image_generation_router.post("/stable")
async def create_stable_image_generation(payload: GenerateStableRequest):
    return {
        "base64_image": image_generation_service.generate_image_stable(
            clothes=payload.clothes,
            person=payload.person
        )
    }
