# This file is modified from https://github.com/haotian-liu/LLaVA/

import re
import cv2

from io import BytesIO

import requests
import torch
from PIL import Image

import base64
import time
from openai import OpenAI
import PIL
import numpy as np

from problem_reduction.vila.prompts import get_prompt

import numpy as np
from problem_reduction.vila.encode import scale_path
from problem_reduction.vila.decode import (
    get_path_from_answer,
    add_mask_2d_to_img,
    add_path_2d_to_img,
    get_mask_2d,
)


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", args.video_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def center_crop_and_resize(
    image: Image.Image, crop_size: int, resize_size: int
) -> Image.Image:
    """Center crops an image to `crop_size` and resizes it to `resize_size`."""
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((resize_size, resize_size), Image.LANCZOS)


# def add_path_2d_to_img(
#     image, points, line_size=1, circle_size=0, plot_lines=True, color="red"
# ):
#     img_out = image.copy()

#     if np.all(points <= 1):
#         points = points * image.shape[1]

#     points = points.astype(int)
#     path_len = len(points)

#     # Generate gradient from dark red to bright red
#     if color == "red":
#         color_choice = np.linspace(25, 255, path_len).astype(int)
#         colors = [tuple(int(r) for r in (r_val, 0, 0)) for r_val in color_choice]
#     # Generate gradient from dark blue to bright blue
#     elif color == "blue":
#         color_choice = np.linspace(25, 255, path_len).astype(int)
#         colors = [tuple(int(r) for r in (0, 0, r_val)) for r_val in color_choice]

#     for i in range(path_len - 1):
#         color = colors[i]
#         if plot_lines:
#             cv2.line(img_out, tuple(points[i]), tuple(points[i + 1]), color, line_size)
#         if circle_size > 0:
#             cv2.circle(
#                 img_out,
#                 tuple(points[i]),
#                 max(1, circle_size),
#                 color,
#                 -1,
#                 lineType=cv2.LINE_AA,
#             )

#     # Draw last point
#     if circle_size > 0:
#         cv2.circle(
#             img_out,
#             tuple(points[-1]),
#             max(1, circle_size),
#             colors[-1],
#             -1,
#             lineType=cv2.LINE_AA,
#         )

#     return img_out

try:
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.conversation import conv_templates
    from llava.utils import disable_torch_init
except:
    print("WARNING: llava installation not found, only API inference available")

def load_model(version, args):
    if version == "vila":
        from llava.model.builder import load_pretrained_model
    elif version == "nvila":
        import llava

    global tokenizer, model, image_processor, context_len

    print("Loading model", version, args.model_path, args.model_base)

    # standard model
    if args.model_base is None:
        disable_torch_init()
        model_name = get_model_name_from_path(args.model_path)
        if version == "nvila":
            model = llava.load(args.model_path)
        elif version == "vila":
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path, model_name, args.model_base
            )
        print("standard", args.model_path, args.model_base)
        try:
            model = torch.compile(model)
        except:
            print("WARNING: torch.compile not available")
    # LoRA
    else:
        disable_torch_init()

        from llava.model.builder import load_pretrained_model
        from peft import PeftModel

        tokenizer, base_model, image_processor, context_len = load_pretrained_model(
            args.model_base, get_model_name_from_path(args.model_base), model_base=None
        )

        model = PeftModel.from_pretrained(base_model, args.model_path)

        model = model.merge_and_unload()
        print("LoRA", args.model_path, args.model_base)


def inference_nvila(message, args):
    outputs = model.generate_content(message)
    return outputs.strip()


def compute_perplexity(output_logits, target_ids):
    """
    Compute perplexity with debugging for numerical issues.
    """
    log_probs = torch.log_softmax(output_logits, dim=-1)

    target_log_probs = torch.gather(
        log_probs, dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)

    avg_log_prob = target_log_probs.mean(dim=-1)

    perplexity = torch.exp(-avg_log_prob)

    return perplexity


def inference_vila(message, args):

    from llava.constants import IMAGE_TOKEN_INDEX

    quest = message[0]
    images = message[1:]

    # conversation template
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], quest)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # preprocessing
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            # do_sample=True if args.temperature > 0 else False,
            # temperature=args.temperature,
            # top_p=args.top_p,
            # num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    # postprocess
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return outputs.strip()

def inference_vila_batch(messages, args):

    from llava.constants import IMAGE_TOKEN_INDEX

    # messages: list of [quest, PIL.Image] (one image per sample); extend to multi-image as needed
    input_ids_list, images_list = [], []

    # BUILD PROMPTS
    prompts = []
    imgs = []
    for message in messages:
        quest, img = message[0], message[1:]  # imgs is a list of PIL Images

        # build prompt per sample
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], quest)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)
        imgs.append(img)

    # Process images for batch
    imgs_tensor = []
    for img in imgs:
        images_tensor = process_images(img, image_processor, model.config).to(
            model.device, dtype=torch.float16
        )
        imgs_tensor.append(images_tensor)
    imgs_tensor = torch.stack(imgs_tensor, dim=0)

    # Tokenize all prompts
    batch_input_ids = [
        tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .squeeze(0)
        .to(model.device)
        for prompt in prompts
    ]

    # Pad sequences to same length for batching
    max_len = max([len(seq) for seq in batch_input_ids])
    padded_input_ids = []
    for seq in batch_input_ids:
        if len(seq) >= max_len:
            padded_seq = seq[:max_len]
        else:
            pad_token_id = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else 0
            )
            padding = torch.full(
                (max_len - len(seq),),
                pad_token_id,
                dtype=seq.dtype,
                device=seq.device,
            )
            padded_seq = torch.cat([seq, padding])
        padded_input_ids.append(padded_seq)

    batch_input_ids = torch.stack(padded_input_ids)
    
    # ensure padding is defined (helps decoding if you later batch tensors)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            batch_input_ids,                  # list of 1xT tensors
            images=imgs_tensor,              # list of image tensors aligned to inputs
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [o.strip() for o in outputs]
    

def inference_vila_perplexity(message, args, target):

    from llava.constants import IMAGE_TOKEN_INDEX

    quest = message[0]
    images = message[1:]

    # conversation template
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], quest)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # preprocessing
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # inference
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=[images_tensor],
            # do_sample=True if args.temperature > 0 else False,
            # temperature=args.temperature,
            # top_p=args.top_p,
            # num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            # to compute metrics
            return_dict_in_generate=True,
            output_scores=True,
        )

    # postprocess
    output_logits = torch.cat(outputs.scores)[None]
    output_ids = outputs.sequences
    output_string = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
        0
    ].strip()

    split_tag = "TRAJECTORY"
    target_ids = None
    target_lang = target.split(split_tag)[0].strip()
    target_ids = tokenizer(target_lang, return_tensors="pt").input_ids.cuda()[:, 1:]

    # find language prediction in output
    full_output_tokens = output_ids[0].cpu().numpy()
    end_tag = tokenizer.encode(split_tag, add_special_tokens=False)
    end_pos = None
    for i in range(0, len(full_output_tokens) - len(end_tag) + 1):
        if list(full_output_tokens[i : i + len(end_tag)]) == end_tag:
            end_pos = i
            break

    # compute perplexity
    if end_pos is not None:
        answer_logits = output_logits[:, :end_pos, :]
        answer_ids = output_ids[:, :end_pos]
        answer_lang = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[
            0
        ].strip()

        min_len = min(answer_logits.shape[1], target_ids.shape[1])
        perplexity = compute_perplexity(
            answer_logits[:, :min_len, :], target_ids[:, :min_len]
        )

        # regex get string between <quest> and </quest>
        quest = re.search(r"<quest>(.*?)</quest>", quest).group(1)
        print(
            f"QUEST: {quest}\nANSWER: {answer_lang}\nTARGET: {target_lang}\nPERPLEXITY: {perplexity.item():.3f}"
        )

    return output_string


def add_answer_to_img(img, answer, prompt_type, color="red", line_size=3, add_mask=True, mask_pixels=10):

    out = get_path_from_answer(answer, prompt_type)

    h, w, c = img.shape

    # scale path to image size
    mask = None
    scaled_mask = None
    if "mask" in prompt_type:
        min_in, max_in = np.zeros(2), np.array([w, h])
        min_out, max_out = np.zeros(2), np.ones(2)
        mask = out[1] if len(out) == 2 else out
        scaled_mask = scale_path(
            mask, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in
        )

    path = None
    scaled_path = None
    if "path" in prompt_type:
        min_in, max_in = np.zeros(2), np.array([w, h])
        min_out, max_out = np.zeros(2), np.ones(2)
        path = out[0] if len(out) == 2 else out
        scaled_path = scale_path(
            path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in
        )

    if "mask" in prompt_type and scaled_mask is not None and add_mask:
        img = add_mask_2d_to_img(img, scaled_mask, mask_pixels=mask_pixels)
    # mask = get_mask_2d(img, scaled_mask, mask_pixels=mask_pixels)

    if "path" in prompt_type and scaled_path is not None:
        img = add_path_2d_to_img(img, scaled_path)

    return img, path, mask


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
            prompt = get_prompt(quest, prompt_type)
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


def vila_inference(rgb, lang_instr, prompt_type="path_mask", args={}):

    prompt = get_prompt(quest=lang_instr, prompt_type=prompt_type)
    image = Image.fromarray(rgb)

    # preprocess
    image = center_crop_and_resize(image, min(image.size), 384)
    message = [prompt, image]

    # inference
    answer_pred = inference_vila(message, args)

    # postprocess
    image = np.array(image)

    image, path_pred, mask_pred = add_answer_to_img(
        image, answer_pred, prompt_type, color="red", add_mask=True
    )

    return image, path_pred, mask_pred

def vila_inference_batch(rgbs, lang_instrs, prompt_type="path_mask", args={}):

    messages = []

    for rgb, lang_instr in zip(rgbs, lang_instrs):
        prompt = get_prompt(quest=lang_instr, prompt_type=prompt_type)
        image_raw = Image.fromarray(rgb)

        # preprocess
        image_cropped = center_crop_and_resize(image_raw, min(image_raw.size), 384)
        message = [prompt, image_cropped]
        messages.append(message)

    # inference
    answer_preds = inference_vila_batch(messages, args)
    
    # postprocess
    image_preds = []
    path_preds = []
    mask_preds = []

    for rgb, answer_pred in zip(rgbs, answer_preds):
        image_raw = np.array(rgb)
        image_pred, path_pred, mask_pred = add_answer_to_img(
            image_raw, answer_pred, prompt_type, color="red", add_mask=True, mask_pixels=10
        )
        image_preds.append(image_pred)
        path_preds.append(path_pred)
        mask_preds.append(mask_pred)

    return image_preds, path_preds, mask_preds

def vila_inference_api(rgb, lang_instr, model_name, server_ip, prompt_type="path_mask"):

    answer_pred = send_request(rgb, lang_instr, prompt_type=prompt_type, server_ip=server_ip, model_name=model_name)

    image, path_pred, mask_pred = add_answer_to_img(
        rgb, answer_pred, prompt_type, color="red", add_mask=True
    )

    return image, path_pred, mask_pred

def hamster_inference_api(rgb, lang_instr, model_name, server_ip, prompt_type="hamster"):

    assert model_name == "hamster_13b"
    answer_pred = send_request(rgb, lang_instr, prompt_type=prompt_type, server_ip=server_ip, model_name=model_name)

    from problem_reduction.vila.inference_hamster import process_answer, draw_lines_on_image_cv, annotate_image

    response_text_strip = re.search(r'<ans>(.*?)</ans>', answer_pred, re.DOTALL).group(1)
    points = process_answer(response_text_strip)
    output_image = draw_lines_on_image_cv(image.copy(), points, draw_action=True)
    annotated_image = annotate_image(output_image.copy(), quest)

    return annotated_image, points, None


if __name__ == "__main__":
    image_path = "/gscratch/weirdlab/memmelma/simvla/pick_data_gen/house.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    quest = "What is in the image?"
    prompt_type = "path_mask"
    server_ip = "https://ccca-198-48-92-26.ngrok-free.app"
    response = send_request(image, quest, prompt_type, server_ip)
    print(response)
