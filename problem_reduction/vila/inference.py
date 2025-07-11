import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def image_parser(args):
    if "image_file" in args:
        out = [args.image_file.split(args.sep)]
        return out, None
    elif "image_url" in args:
        out = [args.image_url]
        return None, out
    else:
        raise ValueError("No image file or image url provided")

import base64
import re
IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", args.video_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_image_from_url(image_url):
    if image_url.startswith("http") or image_url.startswith("https"):
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url: {image_url}")
        image_base64 = match_results.groups()[1]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    return image

def load_images(image_files=None, image_urls=None):
    out = []
    if image_files:
        for image_file in image_files:
            image = load_image(image_file)
            out.append(image)
    elif image_urls:
        for image_url in image_urls:
            image = load_image_from_url(image_url)
            out.append(image)
    return out


def load_model(args):
    # standard model
    if args.model_base is None:
        disable_torch_init()
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, model_name, args.model_base
        )
        print("standard", args.model_path, args.model_base)
    # LoRA
    else:
        disable_torch_init()
        from peft import PeftModel

        tokenizer, base_model, image_processor, context_len = load_pretrained_model(
            args.model_base, get_model_name_from_path(args.model_base), model_base=None
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)
        model = model.merge_and_unload()
        print("LoRA", args.model_path, args.model_base)
    return model, tokenizer, image_processor, context_len


def inference(model, tokenizer, image_processor, args, resize_crop=False):

    # load images
    image_files, image_urls = image_parser(args)
    images = load_images(image_files, image_urls)
    
    if resize_crop:

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
            return cropped.resize((resize_size, resize_size), Image.BILINEAR)

        images = [center_crop_and_resize(img, min(img.size), 384) for img in images]

    # construct prompt
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            print(
                "no <image> tag found in input. Automatically append one at the beginning of text."
            )
            # do not repeatively append the prompt.
            if model.config.mm_use_im_start_end:
                qs = (image_token_se + "\n") * len(images) + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print("query: ", qs)
    print("prompt: ", prompt)

    # preprocessing
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # postprocess
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print("outputs: ", outputs)

    return images, prompt, outputs
