import os
import io
import base64
from abc import ABC, abstractmethod
from fastapi import HTTPException
from PIL import Image
from rembg import remove

class IBackgroundRemovalService(ABC):
    @abstractmethod
    def remove_background(self, photo_path: str) -> str:
        pass

class BackgroundRemovalService(IBackgroundRemovalService):
    def remove_background(self, photo_path: str) -> str:

        if not os.path.exists(photo_path):
            raise HTTPException(status_code=404, detail="File not found")

        try:
            img = Image.open(photo_path).convert("RGBA")
            no_bg = remove(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Background removal failed: {e}")

        try:
            buffer = io.BytesIO()
            no_bg.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{b64}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to encode image: {e}")

