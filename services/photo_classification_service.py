import os
import io
import json
import base64
import re
from abc import ABC, abstractmethod
from fastapi import HTTPException
from PIL import Image, ImageFilter
from openai import OpenAI

client = OpenAI(api_key="")


class IPhotoClassificationService(ABC):
    @abstractmethod
    def classify_photo(self, photo_path: str) -> dict:
        pass


class PhotoClassificationService(IPhotoClassificationService):
    PROMPT = """
    You are a highly meticulous fashion analyst. Take your time to **carefully examine** every detail of the image—lighting, texture, color gradients, stitching and silhouette—before composing your answer. Ensure you consider the **visual clarity** and **fine details** to make the most accurate classification.

    I will provide you with an image of a piece of clothing. Please analyze it and return a JSON object with exactly the following fields (matching the Java PieceOfClothes class):

    {
      "name": string,
      "color": string,
      "material": string,
      "styles": [ "Sporty","Casual","Business","Evening" ],
      "pieceCategory": "Single"|"Outerlayer"|"Innerlayer"|"Bottom"|"Headwear"|"Footwear"|"Accessories",
      "temperatureCategories": [
        "ExtremeHeat","Hot","Warm","MildWarm","Cool","Cold",
        "LightFrost","ChillyFrost","Frost","SevereFrost","ExtremeCold"
      ],
      "characteristics": [string]
    }

    Use these precise temperature mappings:
    - ExtremeHeat: ≥30°C
    - Hot: 25–29°C
    - Warm: 20–24°C
    - MildWarm: 15–19°C
    - Cool: 10–14°C
    - Cold: 5–9°C
    - LightFrost: 0–4°C
    - ChillyFrost: -5–-1°C
    - Frost: -10–-6°C
    - SevereFrost: -19–-11°C
    - ExtremeCold: ≤-20°C

    Define pieceCategory as:
    - Single: one-piece garment (e.g. dress, jumper)
    - Outerlayer: worn over other clothes (e.g. jacket, coat)
    - Innerlayer: garments worn directly on the body (e.g. t-shirts, shirts, sweaters, tank tops)
    - Bottom: lower-body wear (e.g. pants, skirt, shorts)
    - Headwear: items worn on the head (e.g. hat, cap, beanie)
    - Footwear: shoes, boots, sandals
    - Accessories: non-garment items (e.g. scarf, belt, gloves, jewelry)

    Constraints:
    - "color" is only the dominant color.
    - "characteristics" may include patterns, textures, secondary colors, etc.
    - Return only the JSON object—**no extra text**.
    - Answer in Ukrainian.
    """.strip()

    def classify_photo(self, photo_path: str) -> dict:
        if not os.path.isfile(photo_path):
            raise HTTPException(status_code=404, detail="Image file not found")

        try:
            image = Image.open(photo_path).convert("RGB")
            image.thumbnail((600, 600), Image.Resampling.LANCZOS)
            image = image.filter(ImageFilter.MedianFilter(size=3))

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90, subsampling=0)
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies clothing from images."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI API request failed: {e}")

        raw_content = response.choices[0].message.content.strip()
        cleaned = self._extract_json_block(raw_content)

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse JSON: {e}\nRaw response:\n{cleaned}"
            )

        expected_keys = {
            "name", "color", "material", "styles",
            "pieceCategory", "temperatureCategories", "characteristics"
        }

        missing_keys = expected_keys - result.keys()
        if missing_keys:
            raise HTTPException(status_code=500, detail=f"Missing keys in response: {missing_keys}")

        return {
            "name": result["name"],
            "color": result["color"],
            "material": result["material"],
            "styles": result["styles"],
            "pieceCategory": result["pieceCategory"],
            "temperatureCategories": result["temperatureCategories"],
            "characteristics": result["characteristics"]
        }

    def _extract_json_block(self, text: str) -> str:

        fenced_match = re.match(r"^```(?:json)?\s*(\{.*?\})\s*```$", text, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)


        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end + 1]

        return text
