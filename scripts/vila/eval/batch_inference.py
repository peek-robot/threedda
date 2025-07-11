import os
import json
import cv2
import argparse
import numpy as np
from PIL import Image

from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from problem_reduction.vila.inference_helpers import load_model, inference_vila, inference_nvila, center_crop_and_resize, add_answer_to_img

def run_batch_inference(args, prompt_type, model_args):

    # load model
    version = "nvila" if "nvila" in args.model_name else "vila"
    load_model(version, model_args)

    save_path = os.path.join(args.out_path, args.model_name)
    os.makedirs(save_path, exist_ok=True)

    data = []
    with open(args.json_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    # with open(args.json_path, "r") as f:
    #     data = json.load(f)

    data = data[:args.num_samples]

    logs = {
        "img_path": [],
        "path_pred": [],
        "path_gt": [],
        "dtw_euclidian": [],
        "first_l2": [],
        "last_l2": [],
        "iou": [],
    }
    for qa in tqdm(data):
        
        # load sample
        img_path = qa["image"]
        answer_gt = qa["conversations"][1]["value"]
        prompt = qa["conversations"][0]["value"]
        image = load_image(img_path)

        # preprocess
        image = center_crop_and_resize(image, min(image.size), 384)
        message = [prompt, image]

        # inference
        if version == "vila":
            answer_pred = inference_vila(message, model_args)
            # answer_pred = inference_vila_perplexity(message, model_args, target=answer_gt if "INSTRUCTION" in answer_gt else None)
        elif version == "nvila":
            answer_pred = inference_nvila(message, model_args)

        # postprocess
        image = np.array(image)
        try:
            image, path_pred, mask_pred = add_answer_to_img(image, answer_pred, prompt_type, color="red", add_mask=True)
            image, path_gt, mask_gt = add_answer_to_img(image, answer_gt, prompt_type, color="blue", add_mask=False)
        except Exception as e:
            print(f"ERROR: {img_path} failed with '{e}'!")
            continue

        def compute_iou(mask1, mask2):
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            return intersection / union if union != 0 else 0.0

        def plot_masks(mask1, mask2):
            h, w = mask1.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
            # red for prediction
            img[..., 0] = mask1 * 255
            # blue for ground truth
            img[..., 2] = mask2 * 255
            # green for overlapping regions
            overlap = np.logical_and(mask1, mask2)
            img[overlap] = [0, 255, 0]
            return img

        # compute metrics
        iou = compute_iou(mask_pred, mask_gt)
        mask = plot_masks(mask_pred, mask_gt)
        # add metrics to mask
        text = f"IoU {iou:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(mask, (8, mask.shape[0]-40), (8 + text_width + 4, mask.shape[0]-10), (255,255,255), -1)
        cv2.putText(mask, text, (10, mask.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

        # compute metrics
        dtw_euclidian, dtw_path = fastdtw(path_gt, path_pred, dist=euclidean)
        dtw_euclidian = dtw_euclidian / len(path_gt)
        first_l2 = np.linalg.norm(path_gt[0] - path_pred[0])
        last_l2 = np.linalg.norm(path_gt[-1] - path_pred[-1])
        # add metrics to image
        text = f"DTW {dtw_euclidian:.2f} first {first_l2:.2f} last {last_l2:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(image, (8, image.shape[0]-40), (8 + text_width + 4, image.shape[0]-10), (255,255,255), -1)
        cv2.putText(image, text, (10, image.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

        # add legend
        cv2.putText(image, "predicted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
        cv2.putText(image, "ground truth", (image.shape[1]-160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        # save image
        img_path = os.path.join(save_path, f"{img_path.split('/')[-1]}.png")
        Image.fromarray(np.concatenate([image, mask], axis=1)).save(img_path)

        # log results
        logs["img_path"].append(img_path)
        logs["path_pred"].append(path_pred)
        logs["path_gt"].append(path_gt)
        logs["dtw_euclidian"].append(dtw_euclidian)
        logs["first_l2"].append(first_l2)
        logs["last_l2"].append(last_l2)
        logs["iou"].append(iou)

    # dump logs to json
    with open(os.path.join(save_path, "results.json"), "w") as f:
        results = []
        for i in range(len(logs["img_path"])):
            results.append({
                "img_path": logs["img_path"][i],
                "path_pred": logs["path_pred"][i].tolist(),
                "path_gt": logs["path_gt"][i].tolist(),
                "dtw_euclidian": logs["dtw_euclidian"][i],
                "first_l2": logs["first_l2"][i],
                "last_l2": logs["last_l2"][i],
                "iou": logs["iou"][i],
            })
        json.dump(results, f)

    # dump summary to json
    summary = {
        "dtw_euclidian": np.mean(logs["dtw_euclidian"]),
        "first_l2": np.mean(logs["first_l2"]),
        "last_l2": np.mean(logs["last_l2"]),
        "iou": np.mean(logs["iou"]),
    }
    with open(os.path.join(save_path, "summary.json"), "w") as f:
        json.dump(summary, f)

    return summary

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)

    parser.add_argument("--model_path", type=str, default="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/checkpoints/finetuned/nvila/")
    parser.add_argument("--out_path", type=str, default="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/results")

    parser.add_argument("--prompt_type", type=str, choices=["path", "mask", "path_mask"], default="path")
    parser.add_argument("--num_samples", type=int, default=-1)

    args = parser.parse_args()

    # set these for LoRA
    base_name, base_path = None, None
    
    args_dict = {
        "model_path": args.model_name if args.model_path is None else os.path.join(args.model_path, args.model_name),
        "conv_mode": "vicuna_v1",
        "model_base": base_name if base_path is None else os.path.join(base_path, base_name),
        # "temperature": 0.2,
        # "top_p": None,
        # "num_beams": 1,
        "max_new_tokens": 1024,
    }
    model_args = argparse.Namespace(**args_dict)

    summary = run_batch_inference(args, prompt_type=args.prompt_type, model_args=model_args)
    print("Summary", summary)
    
    # python batch_inference.py --model_name nvila_lite_2b_oxe_robopoint --json_path /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data/oxe_processed_path_subtraj/bridge_v2_primary_path/train_bridge_v2_primary_path_conv.json --prompt_type path --num_samples 100
    # python batch_inference.py --model_name nvila_lite_8b_oxe_robopoint --json_path /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data/oxe_processed_path_subtraj/bridge_v2_primary_path/train_bridge_v2_primary_path_conv.json --prompt_type path --num_samples 100
    # python batch_inference.py --model_name vila_3b_oxe_robopoint --json_path /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data/oxe_processed_path_subtraj/bridge_v2_primary_path/train_bridge_v2_primary_path_conv.json --prompt_type path --num_samples 100
    # python batch_inference.py --model_name vila_13b_oxe_robopoint --json_path /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data/oxe_processed_path_subtraj/bridge_v2_primary_path/train_bridge_v2_primary_path_conv.json --prompt_type path --num_samples 100