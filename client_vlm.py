import base64
import time
import cv2
from openai import OpenAI
import PIL
import numpy as np

import sys
sys.path.append("/mmfs1/gscratch/weirdlab/memmelma/simvla/vila_utils/")
from vila_utils.utils.prompts import get_prompt
from scripts.inference_helpers import center_crop_and_resize

MODEL_NAME = "vila_13b_path_mask_new"

def send_request(
    image,
    quest,
    prompt_type,
    server_ip,
    max_tokens=1024,
    temperature=0.0,
    top_p=0.95,
    max_retries=5,
    verbose=False,
):
    """Send image and quest to HAMSTER model and get response."""
    # Ensure image is in BGR format for OpenCV
    
    image = PIL.Image.fromarray(image)
    # preprocess the image
    image_resized = center_crop_and_resize(image, min(image.size), 384)

    # Encode image to base64
    image_resized = np.asarray(image_resized)
    _, encoded_image_array = cv2.imencode(".jpg", image_resized)
    encoded_image = base64.b64encode(encoded_image_array.tobytes()).decode("utf-8")

    if verbose:
        print(f"Sending request with quest: {quest}")

    retry_count = 0
    while retry_count < max_retries:
        try:
            start_time = time.time()  # Record start time
            client = OpenAI(base_url=server_ip, api_key="fake-key")
            prompt = get_prompt(quest, prompt_type)
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=int(max_tokens),
                model=MODEL_NAME,
                extra_body={
                    "prompt_type": prompt_type,
                    "num_beams": 1,
                    "use_cache": True,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                },
            )
            end_time = time.time()  # Record end time
            response_text = response.choices[0].message.content[0]["text"]
            duration = end_time - start_time
            if verbose:
                print(f"Server response received in {duration:.2f} seconds.")
            return response_text
        except Exception as e:
            retry_count += 1
            wait_time = 2**retry_count  # Exponential backoff
            if retry_count < max_retries:
                print(f"Error connecting to server: {e}")
                print(f"Retrying in {wait_time} seconds... (Attempt {retry_count} of {max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
    return None


if __name__ == "__main__":
    image_path = "/gscratch/weirdlab/memmelma/simvla/pick_data_gen/house.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    quest = "What is in the image?"
    prompt_type = "path_mask"
    server_ip = "https://ccca-198-48-92-26.ngrok-free.app"
    response = send_request(image, quest, prompt_type,server_ip)
    print(response)