import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

from fastapi import HTTPException
import requests


class IImageGenerationService(ABC):
    @abstractmethod
    def generate_image(self, image_paths: List[str],
        person_payload: Union[str, Dict[str, Any]],) -> str:
        pass

    @abstractmethod
    def generate_image_stable(self, clothes, person):
        pass

class ImageGenerationService(IImageGenerationService):

    _ENDPOINT = "https://api.openai.com/v1/images/edits"
    _MODEL = "gpt-image-1"
    _HEADERS = {
        "Authorization": "Bearer ",
    }

    def generate_image(
        self,
        image_paths: List[str],
        person_payload: Union[str, Dict[str, Any]],
    ) -> str:
        if not image_paths:
            raise HTTPException(status_code=400, detail="image_paths must not be empty")

        files = []
        prompt = ""

        if isinstance(person_payload, str):
            person_path = Path(person_payload)
            if not person_path.is_file():
                raise HTTPException(status_code=404, detail=f"Person image not found: {person_path}")
            files.append(("image[]", (person_path.name, person_path.read_bytes(), "image/png")))
            prompt = (
                "You are a visual assistant specializing in generating photorealistic images of people dressed in different clothing."
                "Your task is to replace the outfit of the person in the first image entirely, using only their visible pose, body shape, and skin tone as reference. "
                "Disregard their original clothing."
                "Use the garments provided in the reference images to dress the person. "
                "Ensure that the final image shows the individual in full height, from head to toe — including their shoes, which must be clearly visible. "
                "If needed, create a bit more distance from the subject to properly frame the full body and preserve the integrity of the shoes and lower garments."
                "Take your time to ensure high visual quality, anatomical accuracy, and seamless integration of the new outfit. "
                "Reconstruct any hidden or unclear body parts naturally, based on realistic human proportions. "
                "Preserve the person`s original pose, lighting, and overall realism throughout."
                
            )
        elif isinstance(person_payload, dict):
            prompt = (
                f"{self._describe_person(person_payload)}"
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid person_payload format")

        for path in image_paths:
            p = Path(path)
            if not p.is_file():
                raise HTTPException(status_code=404, detail=f"Image not found: {p}")
            files.append(("image[]", (p.name, p.read_bytes(), "image/png")))

        data = {
            "model": self._MODEL,
            "prompt": prompt,
        }

        try:
            response = requests.post(
                self._ENDPOINT,
                headers=self._HEADERS,
                data=data,
                files=files,
                timeout=300,
            )
            response.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}")

        try:
            result = response.json()
            base64_image = result["data"][0]["b64_json"]

            return base64_image
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to decode image: {e}")

    @staticmethod
    def _describe_person(info: Dict[str, Any]) -> str:
        return (
            "Person description:\n"
            f"- gender: {info.get('gender', 'unspecified')}\n"
            f"- skin tone: {info.get('skinTone', 'unspecified')}\n"
            f"- hair color: {info.get('hairColor', 'unspecified')}\n"
            f"- height: {info.get('height', 'unknown')} cm\n"
            f"- age: {info.get('age', 'unknown')} years"
        )

    def generate_image_stable(self, clothes, person):
        def format_person(p):
            return f"{p.get('gender', 'person')}, hair: {p.get('hairColor', '')}, height: {p.get('height', '')} cm, age: {p.get('age', '')} years, skin tone: {p.get('skinTone', '')}"

        def format_clothing_piece(piece):
            attrs = []
            if piece.get("color"):
                attrs.append(piece["color"])
            if piece.get("material"):
                attrs.append(piece["material"])
            if piece.get("name"):
                attrs.append(piece["name"])
            if piece.get("characteristics"):
                attrs.extend(piece["characteristics"])
            return " ".join(attrs)

        person_dict = person.dict()

        person_desc = format_person(person_dict)

        clothes_dicts = [c.dict() for c in clothes]

        clothes_desc = ", ".join(format_clothing_piece(c) for c in clothes_dicts)

        prompt = (
            f"A full-body photorealistic render of {person_desc}, wearing: {clothes_desc}. "
            "High detail, natural lighting, realistic proportions, detailed face"
            "straight pose"
        )

        negative_prompt = (
            "blurry, low quality, cropped, deformed body, disfigured face, extra limbs, "
            "text, watermark, logo, lowres, jpeg artifacts, cartoon, unrealistic, duplicate"
        )

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampler_name": "DPM++ 2M Karras",
            "steps": 23,
            "cfg_scale": 7,
            "width": 512,
            "height": 768,
            "restore_faces": True,
            "tiling": False,
            "batch_size": 1,
            "n_iter": 1,
            "enable_hr": True,
            "hr_scale": 1.5,
            "hr_upscaler": "Latent",
            "hr_second_pass_steps": 5,
            "send_images": True,
            "save_images": False,
            "denoising_strength": 0.4
        }

        try:

            response = requests.post(
                "http://127.0.0.1:7860/sdapi/v1/txt2img",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=600
            )
            response.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Stable Diffusion request failed: {e}")

        try:
            result = response.json()

            base64_image = result["images"][0]

            return base64_image
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to decode image: {e}")

