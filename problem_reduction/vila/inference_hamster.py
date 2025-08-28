import cv2
import numpy as np
from matplotlib import cm
import PIL
import base64
import time
import re
from io import BytesIO
from openai import OpenAI
from problem_reduction.vila.prompts import get_prompt
from problem_reduction.vila.inference_helpers import center_crop_and_resize

GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1

def annotate_image(image, quest):
    """
    Annotate the given image by overlaying the quest (prompt) text in the top-left corner,
    then save the image with a timestamp in the filename.
    
    The image is assumed to be in BGR color space.
    """
    # Choose font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Get size of the text box
    (text_w, text_h), baseline = cv2.getTextSize(quest, font, font_scale, thickness)
    # Draw a filled rectangle as background for the text for better visibility
    cv2.rectangle(image, (5, 5), (5 + text_w + 10, 5 + text_h + 10), (0, 0, 0), -1)
    # Overlay the quest text on top of the rectangle
    cv2.putText(image, quest, (10, 5 + text_h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return image

def draw_lines_on_image_cv(image, points, draw_action=False, num_subdivisions=100):
    height, width, _ = image.shape

    # Calculate a scale factor relative to a 256x256 image -> original was 512x512 but w/ 128x128 gripper actions is not visible
    scale_factor = min(min(width, height) / 256.0, 1)
    circle_radius = int(7 * scale_factor)
    circle_thickness = max(1, int(2 * scale_factor))
    line_thickness = max(1, int(2 * scale_factor))
    font_scale = 0.5 * scale_factor
    font_thickness = max(1, int(1 * scale_factor))
    text_color = (255, 255, 255)  # White color

    # Convert normalized coordinates to pixel coordinates
    pixel_points = []
    gripper_status = []
    for point in points:
        x = int(point[0] * width)
        y = int(point[1] * height)
        action = int(point[2])
        pixel_points.append((x, y))
        gripper_status.append(action)

    # Draw optional markers or numbers at the predicted points
    for idx, (x, y) in enumerate(pixel_points):
        if draw_action:
            if idx == 0 or gripper_status[idx] != gripper_status[idx - 1]:
                circle_color = (0, 0, 255) if gripper_status[idx] == GRIPPER_CLOSE else (255, 0, 0)
                cv2.circle(image, (x, y), circle_radius, circle_color, circle_thickness)

    # Convert list to NumPy array for interpolation
    pixel_points = np.array(pixel_points, dtype=np.float32)

    # Calculate cumulative distances along the path
    distances = [0]
    for i in range(1, len(pixel_points)):
        dist = np.linalg.norm(pixel_points[i] - pixel_points[i - 1])
        distances.append(distances[-1] + dist)
    total_distance = distances[-1]

    # Generate equally spaced distances along the path
    num_samples = num_subdivisions
    sample_distances = np.linspace(0, total_distance, num_samples)

    # Interpolate points along the path
    interpolated_points = []
    idx = 0
    for sd in sample_distances:
        while sd > distances[idx + 1] and idx < len(distances) - 2:
            idx += 1
        t = (sd - distances[idx]) / (distances[idx + 1] - distances[idx])
        point = (1 - t) * pixel_points[idx] + t * pixel_points[idx + 1]
        interpolated_points.append(point)
    interpolated_points = np.array(interpolated_points, dtype=np.int32)

    # Map positions along the path to colors using the jet colormap
    cmap = cm.get_cmap('jet')
    colors = (cmap(np.linspace(0, 1, len(interpolated_points)))[:, :3] * 255).astype(np.uint8)

    # Draw line segments with varying colors using the scaled line thickness
    for i in range(len(interpolated_points) - 1):
        pt1 = tuple(interpolated_points[i])
        pt2 = tuple(interpolated_points[i + 1])
        color = tuple(int(c) for c in colors[i])
        cv2.line(image, pt1, pt2, color=color, thickness=line_thickness)

    return image

def process_answer(input_str):
    """Extract keypoints from the model response."""
    input_str = input_str.replace('<action>Close Gripper</action>', '(1000.0, 1000.0)').replace('<action>Open Gripper</action>', '(1001.0, 1001.0)')
    keypoints = eval(input_str)
    processed_points = []
    action_flag = 0
    for point in keypoints:
        x, y = point
        if x == y and x == 1000.0:
            action_flag = GRIPPER_CLOSE
            processed_points[-1][-1] = action_flag
            continue
        elif x == y and x == 1001.0:
            action_flag = GRIPPER_OPEN
            processed_points[-1][-1] = action_flag
            continue
        processed_points.append([x, y, action_flag])
    return processed_points


def send_request(
    image,
    quest,
    prompt_type,
    server_ip,
    max_tokens=1024,
    temperature=0.0,
    top_p=0.95,
    max_retries=5,
    model_name="vila_13b_path_mask_new",
    verbose=False,
):
    """Send image and quest to HAMSTER model and get response."""
    # Ensure image is in BGR format for OpenCV

    image = PIL.Image.fromarray(image)
    # preprocess the image
    image_resized = center_crop_and_resize(image, min(image.size), 384)

    # Encode image to base64
    image_resized = np.asarray(image_resized)
    # _, encoded_image_array = cv2.imencode(".jpg", image_resized)
    # encoded_image = base64.b64encode(encoded_image_array.tobytes()).decode("utf-8")

    buffer = BytesIO()
    PIL.Image.fromarray(image_resized).save(buffer, format='JPEG')
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    if verbose:
        print(f"Sending request with quest: {quest}")

    retry_count = 0
    while retry_count < max_retries:
        try:
            start_time = time.time()  # Record start time
            client = OpenAI(base_url=server_ip, api_key="fake-key")
            prompt = get_prompt(quest, prompt_type, prompt_eval=False)
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=int(max_tokens),
                model=model_name,
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
                print(
                    f"Retrying in {wait_time} seconds... (Attempt {retry_count} of {max_retries})"
                )
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
    return None

def hamster_inference_api(rgb, lang_instr, model_name, server_ip, prompt_type="hamster"):

    assert model_name == "hamster_13b"
    answer_pred = send_request(rgb, lang_instr, prompt_type=prompt_type, server_ip=server_ip, model_name=model_name)

    from problem_reduction.vila.inference_hamster import process_answer, draw_lines_on_image_cv, annotate_image

    response_text_strip = re.search(r'<ans>(.*?)</ans>', answer_pred, re.DOTALL).group(1)
    points = process_answer(response_text_strip)
    output_image = draw_lines_on_image_cv(rgb.copy(), points, draw_action=True)
    # annotated_image = annotate_image(output_image.copy(), lang_instr)

    return output_image, points