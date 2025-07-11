import argparse
import os
import time
import uuid
from contextlib import asynccontextmanager
import re
import base64
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from PIL import Image
from llava.utils import disable_torch_init

from io import BytesIO
from pydantic import BaseModel
from typing import List, Literal, Optional, Union, get_args

import sys
sys.path.append("/mmfs1/gscratch/weirdlab/memmelma/simvla/vila_utils/")
from scripts.inference_helpers import center_crop_and_resize, inference_vila, load_model, add_answer_to_img
from vila_utils.utils.prompts import get_prompt

tokenizer = None
model = None
image_processor = None
context_len = None

class TextContent(BaseModel):
    type: Literal["text"]
    text: str

class ImageURL(BaseModel):
    url: str
    
class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]
    
class ChatCompletionRequest(BaseModel):
    model: Literal["vila_13b_path_mask_new"]
    prompt_type: Literal["path_mask", "robotpoint"]
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    use_cache: Optional[bool] = True
    num_beams: Optional[int] = 1

IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")

def load_image(image_url: str) -> Image:
    if image_url.startswith("http") or image_url.startswith("https"):
        import requests
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url: {image_url}")
        image_base64 = match_results.groups()[1]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    return image

@asynccontextmanager
async def lifespan(app: FastAPI):

    global tokenizer, model, image_processor, context_len

    disable_torch_init()
    
    args_dict = {
        "model_path": app.args.model_path,
        "conv_mode": "vicuna_v1",
        "model_base": None,
        # "temperature": 0.2,
        # "top_p": None,
        # "num_beams": 1,
        "max_new_tokens": 1024,
    }
    model_args = argparse.Namespace(**args_dict)
    load_model("vila", model_args)

    yield


app = FastAPI(lifespan=lifespan)


# Load model upon startup
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        global tokenizer, model, image_processor, context_len

        model_args = {
            "max_new_tokens": request.max_tokens,
            "conv_mode": "vicuna_v1",
            # "temperature": request.temperature,
            # "top_p": request.top_p,
            # "num_beams": request.num_beams,
            # "use_cache": request.use_cache,
        }
        model_args = argparse.Namespace(**model_args)
        
        prompt = request.messages[0].content[1].text
        image = load_image(request.messages[0].content[0].image_url.url)
        image = center_crop_and_resize(image, min(image.size), 384)
        message = [prompt, image]

        print("USER:\n", prompt)
        answer_pred = inference_vila(message, model_args)
        print("\nASSISTANT:\n", answer_pred)

        resp_content = [TextContent(type="text", text=answer_pred)]
        return {
            "id": uuid.uuid4().hex,
            "object": "chat.completion",
            "created": time.time(),
            "model": request.model,
            "choices": [{"message": ChatMessage(role="assistant", content=resp_content)}],
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_path", type=str, required=True)
    app.args = parser.parse_args()

    uvicorn.run(app, host=app.args.host, port=app.args.port, workers=1, log_level="debug")