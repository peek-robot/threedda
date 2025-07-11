import os
import re
import ast
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from problem_reduction.vila.slurm import get_slurm_time_left
from problem_reduction.vila.encode import smooth_path_rdp, scale_path
from problem_reduction.vila.prompts import get_prompt, get_answer_from_path, get_answer_from_lang
from problem_reduction.vila.decode import get_path_from_answer, add_path_2d_to_img, add_mask_2d_to_img
from problem_reduction.vila.rewording import RewordLLM

def convert_dataset(task, dataclass, split, data_dir, prompt_type, reword_quest=False, num_samples=np.inf, min_points=3, path_rdp_tolerance=0.05, mask_rdp_tolerance=0.05, save_sketches_every_n=False):
    
    if reword_quest:
        reword_llm = RewordLLM(model_name="meta-llama/Llama-3.1-8B-Instruct", n_samples=5)
        
    # reword_llm.reword("Clean the top of the shelf with a washcloth.")

    entries_conv = []
    entries_vqa = []

    # setup dirs
    data_dir = os.path.join(data_dir, task)
    os.makedirs(data_dir, exist_ok=True)
    img_dir = os.path.join(data_dir, "images")

    file_conv = open(f'{data_dir}/{split}_{task}_conv.jsonl', 'a')
    file_vqa = open(f'{data_dir}/{split}_{task}_vqa.jsonl', 'a')

    if save_sketches_every_n:
        sketch_dir = os.path.join(data_dir, "sketches")
        os.makedirs(os.path.join(sketch_dir), exist_ok=True)

    # resume functionality
    idx_resume = 0
    idx_file = f'{data_dir}/{split}.txt'
    # read idx from file
    if os.path.isfile(idx_file):
        with open(idx_file, 'r') as file:
            idx_resume = int(file.read())

    idcs = dataclass.get_indices()
    pbar = tqdm(total=len(idcs[idx_resume:]))
    metrics = {
        "avg_path_len": [],
        "avg_path_history_len": [],
        "avg_mask_len": []
    }
    for j, idx in enumerate(idcs[idx_resume:]):
        
        # resume functionality
        curr_idx = idx_resume + j
        # update idx in file
        with open(idx_file, 'w') as file:
            file.write(str(curr_idx + 1))
        
        # step tqdm
        pbar.set_description(f"{split} {curr_idx+1}/{len(idcs)}")
        pbar.update(1)
        
        # step dataset
        dataclass.step(idx)

        # process image
        img, img_path = dataclass.save_image(idx, img_dir)
        if img is None:
            # dataclass.step(idx)
            continue
        h,w,c = img.shape

        # process path/mask
        traj_scaled, traj_history_scaled, mask_scaled = None, None, None

        if "path" in prompt_type:
            path = dataclass.load_path(idx)
            min_in, max_in = np.zeros(2), np.array([w,h])
            min_out, max_out = np.zeros(2), np.ones(2)
            traj_scaled = scale_path(path, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)
            # to check if initial path is valid
            if len(traj_scaled) < min_points:
                continue
            traj_scaled = smooth_path_rdp(traj_scaled, tolerance=path_rdp_tolerance)
            # to check if smoothed path is long enough
            if len(traj_scaled) < min_points:
                continue
            answer = get_answer_from_path(traj_scaled)
            metrics["avg_path_len"].append(len(traj_scaled))

        if "history" in prompt_type:
            path_history = dataclass.load_path_history(idx)
            min_in, max_in = np.zeros(2), np.array([w,h])
            min_out, max_out = np.zeros(2), np.ones(2)
            traj_history_scaled = scale_path(path_history, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)
            
            # to check if initial path is valid
            # if len(traj_history_scaled) < min_points:
            #     traj_history_scaled = []
            #     # continue
            try:
                traj_history_scaled = smooth_path_rdp(traj_history_scaled, tolerance=path_rdp_tolerance)
            except:
                traj_history_scaled = []
            # to check if smoothed path is long enough
            # if len(traj_history_scaled) < min_points:
            #     traj_history_scaled = []
            answer = get_answer_from_path(traj_history_scaled)
            metrics["avg_path_history_len"].append(len(traj_history_scaled))

        if "mask" in prompt_type:
            points = dataclass.load_mask(idx)
            min_in, max_in = np.zeros(2), np.array([w,h])
            min_out, max_out = np.zeros(2), np.ones(2)
            
            # # HACK: add path to mask
            # points = np.concatenate([points,path])
            
            mask_scaled = scale_path(points, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)
            
            # to check if initial mask is valid
            if len(mask_scaled) < min_points:
                continue    
            mask_scaled = smooth_path_rdp(mask_scaled, tolerance=mask_rdp_tolerance)
            # to check if smoothed mask is long enough
            if len(mask_scaled) < min_points:
                continue
            answer = get_answer_from_path(mask_scaled)
            metrics["avg_mask_len"].append(len(mask_scaled))

        if "path" in prompt_type and "mask" in prompt_type:
            answer = "TRAJECTORY: " + get_answer_from_path(traj_scaled)
            answer += " MASK: " + get_answer_from_path(mask_scaled)
        
        if "path" in prompt_type and "mask" in prompt_type and "lang" in prompt_type:
            
            lang = dataclass.load_lang(idx)
            answer = "INSTRUCTION: " + get_answer_from_lang(lang)
            
            answer += " TRAJECTORY: " + get_answer_from_path(traj_scaled)
            answer += " MASK: " + get_answer_from_path(mask_scaled)

        # process quest
        quest = dataclass.load_quest(idx)
        quests = [quest]
        if reword_quest:
            reworded_quest = reword_llm.reword(quest)
            print("QUEST", quest, "\nREWORDED", reworded_quest)
            quests.append(reworded_quest)
        
        # step dataset
        # dataclass.step(idx)

        # construct entry id
        entry_id = img_path + "_" + prompt_type

        for quest in quests:
            # construct conversation format (train)
            entry_conv = {
                # https://github.com/NVlabs/VILA/issues/97
                # "image": ["path/to/image", "path/to/image"],
                "image": img_path,
                "id": entry_id,
                "conversations": [
                    {
                        "from": "human",
                        "value": get_prompt(quest, prompt_type=prompt_type, history=get_answer_from_path(traj_history_scaled) if "history" in prompt_type else None, prompt_eval=False),
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            entries_conv.append(entry_conv)

            # constract VQA format (eval)
            entry_vqa = {
                "image": img_path,
                "question_id": entry_id,
                # same as above but w/o <image>\n -> https://github.com/haotian-liu/LLaVA/issues/732#issuecomment-1994616417
                "text": get_prompt(quest, prompt_type=prompt_type, history=get_answer_from_path(traj_history_scaled) if "history" in prompt_type else None, prompt_eval=True),
                "category": task
            }
            entries_vqa.append(entry_vqa)

        # visualize
        if save_sketches_every_n and j % save_sketches_every_n == 0:
            sketch_path = img_path.replace("images", "sketches").replace(".jpg", "_sketch.jpg")
            os.makedirs(os.path.dirname(sketch_path), exist_ok=True)
            sketch = np.array(img)
            # mask size 5% of image
            mask_pixels = int(img.shape[0] * 0.15)
            if prompt_type == "mask":
                mask = get_path_from_answer(answer, prompt_type=prompt_type)
                mask_scaled = scale_path(mask, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
                sketch = add_mask_2d_to_img(sketch, mask_scaled, mask_pixels=mask_pixels)
            elif prompt_type == "path":
                path = get_path_from_answer(answer, prompt_type=prompt_type)
                path_scaled = scale_path(path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
                sketch = add_path_2d_to_img(sketch, path_scaled, color=(255, 0, 0))
            elif prompt_type == "path_mask":
                out = get_path_from_answer(answer, prompt_type=prompt_type)
                # HACK
                # mask_scaled = scale_path(out[1], min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
                # sketch = add_mask_2d_to_img(sketch, mask_scaled, mask_pixels=mask_pixels)
                
                # traj_scaled = scale_path(out[0], min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
                # sketch = add_path_2d_to_img(sketch, traj_scaled, color=(255, 0, 0))

                mask_scaled = scale_path(out[1], min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)                
                traj_scaled = scale_path(out[0], min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
                sketch = add_mask_2d_to_img(sketch, mask_scaled, mask_pixels=mask_pixels)
                # HACK
                sketch = add_mask_2d_to_img(sketch, traj_scaled, mask_pixels=mask_pixels)
                sketch = add_path_2d_to_img(sketch, traj_scaled, color=(255, 0, 0))
            
            elif prompt_type == "path_mask_history":
                out = get_path_from_answer(answer, prompt_type="path_mask")
                # HACK
                mask_scaled = scale_path(out[1], min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
                sketch = add_mask_2d_to_img(sketch, mask_scaled, mask_pixels=mask_pixels)
                
                traj_scaled = scale_path(out[0], min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
                sketch = add_path_2d_to_img(sketch, traj_scaled, color=(255, 0, 0))
                if len(traj_history_scaled) > 0:
                    traj_history_scaled = scale_path(traj_history_scaled, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
                    sketch = add_path_2d_to_img(sketch, traj_history_scaled, color=(0, 0, 255))

            elif prompt_type == "path_mask_lang":
                out = get_path_from_answer(answer, prompt_type=prompt_type)
                mask_scaled = scale_path(out[2], min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
                sketch = add_mask_2d_to_img(sketch, mask_scaled, mask_pixels=mask_pixels)
                
                traj_scaled = scale_path(out[1], min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
                sketch = add_path_2d_to_img(sketch, traj_scaled, color=(255, 0, 0))
                
                lang_path = sketch_path.replace(".jpg", ".txt")
                with open(lang_path, "w") as f:
                    f.write(out[0])
                
            Image.fromarray(sketch).save(sketch_path)
            
            # log metrics for debugging (e.g., rdp_tolerance, ...)
            print(f"avg_path_len: {np.mean(metrics['avg_path_len'])}, avg_mask_len: {np.mean(metrics['avg_mask_len'])}")
        
        if j == num_samples:
            break
        
        # # save conv as .jsonl format for resume
        # with open(f'{data_dir}/{split}_{task}_conv.jsonl', 'a') as file:
        #     file.write(json.dumps(entry_conv) + '\n')

        # # vqa eval expects .jsonl format
        # with open(f'{data_dir}/{split}_{task}_vqa.jsonl', 'a') as file:
        #     file.write(json.dumps(entry_vqa) + '\n')
        file_conv.write(json.dumps(entry_conv) + '\n')
        file_vqa.write(json.dumps(entry_vqa) + '\n')

        # if < 5min left, kill job preemptively to ensure data is saved
        time_left = get_slurm_time_left()
        if time_left and time_left < 300:
            file_conv.close()
            file_vqa.close()
            exit()

    # convert conv .jsonl to .json
    with open(f'{data_dir}/{split}_{task}_conv.jsonl', 'r') as file:
        entries_conv = [json.loads(line) for line in file]
    with open(f'{data_dir}/{split}_{task}_conv.json', 'w') as file:
        json.dump(entries_conv, file)

    # resume functionality
    # remove idx file
    if os.path.isfile(idx_file):
        os.remove(idx_file)