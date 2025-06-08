from fastapi import FastAPI
from controllers.photo_classification_routes import photo_classification_router
from controllers.image_generation_routes import image_generation_router
from controllers.remove_background_routes import remove_bg_router
app = FastAPI(prefix="v1")

app.include_router(photo_classification_router)
app.include_router(image_generation_router)
app.include_router(remove_bg_router)


