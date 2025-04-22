"""
Name   : run_fuzzer_w_instruction.py
Author : ZHIJIE WANG
Time   : 9/10/24
"""

import argparse
import numpy as np
from experiments.model_interface import VLAInterfaceLM
from pathlib import Path
from tqdm import tqdm
import json
import os
from PIL import Image
import shutil

# Setup paths
PACKAGE_DIR = Path(__file__).parent.resolve()

templates = {
    'grasp': [
        "pick [OBJECT]",
        "grab [OBJECT]",
        "can you pick up [OBJECT]",
        "fetch [OBJECT]",
        "get [OBJECT]",
        "lift [OBJECT]",
        "take [OBJECT]",
        "retrieve [OBJECT]",
        "let's pick up [OBJECT]",
        "would you grab [OBJECT]"
    ],
    'move': [
        "Take [OBJECT A] to [OBJECT B]",
        "Bring [OBJECT A] close to [OBJECT B]",
        "Position [OBJECT A] near [OBJECT B]",
        "Move [OBJECT A] closer to [OBJECT B]",
        "Put [OBJECT A] by [OBJECT B]",
        "Place [OBJECT A] near [OBJECT B]",
        "Set [OBJECT A] next to [OBJECT B]",
        "Can you move [OBJECT A] near [OBJECT B]",
        "Shift [OBJECT A] near [OBJECT B]",
        "Let's move [OBJECT A] near [OBJECT B]"
    ],
    'put-on': [
        "place [OBJECT A] on [OBJECT B]",
        "set [OBJECT A] on [OBJECT B]",
        "move [OBJECT A] onto [OBJECT B]",
        "position [OBJECT A] on [OBJECT B]",
        "put [OBJECT A] onto [OBJECT B]",
        "could you put [OBJECT A] on [OBJECT B]",
        "let's put [OBJECT A] on [OBJECT B]",
        "please place [OBJECT A] on [OBJECT B]",
        "can you place [OBJECT A] on [OBJECT B]",
        "would you move [OBJECT A] onto [OBJECT B]"
    ],
    'put-in': [
        "take [OBJECT] into the yellow basket",
        "bring [OBJECT] into the yellow basket",
        "place [OBJECT] in the yellow basket",
        "move [OBJECT] inside the yellow basket",
        "put [OBJECT] inside the yellow basket",
        "drop [OBJECT] into the yellow basket",
        "insert [OBJECT] into the yellow basket",
        "can you put [OBJECT] into the yellow basket",
        "please put [OBJECT] into the yellow basket",
        "let's put [OBJECT] into the yellow basket"
    ]
}


def _get_instruction_obj_name(s):
    # given an object name, process its name to be used for language instruction
    s = s.split('_')
    rm_list = ['opened', 'light', 'generated', 'modified', 'objaverse', 'bridge', 'baked', 'v2']
    cleaned = []
    for w in s:
        if w[-2:] == "cm":
            # object size in object name
            continue
        if w not in rm_list:
            cleaned.append(w)
    return ' '.join(cleaned)


def mutate_instruction(task, options):
    if task == 'grasp':
        return np.random.choice(templates[task]).replace("[OBJECT]", _get_instruction_obj_name(options["model_id"]))
    elif task == 'move':
        return np.random.choice(templates[task]).replace("[OBJECT A]", _get_instruction_obj_name(options["model_ids"][options["source_obj_id"]])).replace("[OBJECT B]", _get_instruction_obj_name(options["model_ids"][options["target_obj_id"]]))
    elif task == 'put-on':
        return np.random.choice(templates[task]).replace("[OBJECT A]", _get_instruction_obj_name(options["model_ids"][0])).replace("[OBJECT B]", _get_instruction_obj_name(options["model_ids"][1]))
    elif task == 'put-in':
        return np.random.choice(templates[task]).replace("[OBJECT]", _get_instruction_obj_name(options["model_ids"][0]))
    else:
        raise NotImplementedError


class StableJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="VLA Fuzzing")
    parser.add_argument('-d', '--data', type=str, help="Testing data")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output path, e.g., folder")
    parser.add_argument('-io', '--image_output', type=str, default=None, help="Image output path, e.g., folder")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random Seed")
    parser.add_argument('-m', '--model', type=str,
                        choices=["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small", "openvla-7b"],
                        default="rt_1_x",
                        help="VLA model")
    parser.add_argument('-r', '--resume', type=bool, default=True, help="Resume from where we left.")

    args = parser.parse_args()

    random_seed = args.seed if args.seed else np.random.randint(0, 4294967295)  # max uint32

    data_path = args.data if args.data else str(PACKAGE_DIR) + "/../data/t-grasp_n-1000_o-3.json"

    dataset_name = data_path.split('/')[-1]

    task_name = None

    if "grasp" in dataset_name:
        task_name = "grasp"
        if 'ycb' in dataset_name:
            vla = VLAInterfaceLM(model_name=args.model, task="google_robot_pick_customizable_ycb")
        else:
            vla = VLAInterfaceLM(model_name=args.model, task="google_robot_pick_customizable")
    elif "move" in dataset_name:
        task_name = "move"
        if 'ycb' in dataset_name:
            vla = VLAInterfaceLM(model_name=args.model, task="google_robot_move_near_customizable_ycb")
        else:
            vla = VLAInterfaceLM(model_name=args.model, task="google_robot_move_near_customizable")
    elif "put-on" in dataset_name:
        task_name = "put-on"
        if 'ycb' in dataset_name:
            vla = VLAInterfaceLM(model_name=args.model, task="widowx_put_on_customizable_ycb")
        else:
            vla = VLAInterfaceLM(model_name=args.model, task="widowx_put_on_customizable")
    elif "put-in" in dataset_name:
        task_name = "put-in"
        if 'ycb' in dataset_name:
            vla = VLAInterfaceLM(model_name=args.model, task="widowx_put_in_customizable_ycb")
        else:
            vla = VLAInterfaceLM(model_name=args.model, task="widowx_put_in_customizable")
    else:
        raise NotImplementedError

    with open(data_path, 'r') as f:
        tasks = json.load(f)

    if args.output:
        result_dir = args.output + data_path.split('/')[-1].split(".")[0]
    else:
        result_dir = str(PACKAGE_DIR) + "/../results/" + data_path.split('/')[-1].split(".")[0]
    result_dir += '-instruction'
    os.makedirs(result_dir, exist_ok=True)
    result_dir += f'/{args.model}_{random_seed}'
    if not args.resume:
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    if args.image_output:
        image_dir = args.image_output + data_path.split('/')[-1].split(".")[0]
        image_dir += '-instruction'
        os.makedirs(image_dir, exist_ok=True)
        image_dir += f'/{args.model}_{random_seed}'
        os.makedirs(image_dir, exist_ok=True)
    else:
        image_dir = None

    for idx in tqdm(range(tasks["num"])):
        if args.resume and os.path.exists(result_dir + f"/{idx}/" + '/log.json'):  # if resume allowed then skip the finished runs.
            continue
        options = tasks[str(idx)]
        options['task_instruction'] = mutate_instruction(task_name, options)
        print(options["task_instruction"])
        images, episode_stats = vla.run_interface(seed=random_seed, options=options, instruction=options["task_instruction"])
        os.makedirs(result_dir + f"/{idx}", exist_ok=True)
        with open(result_dir + f"/{idx}/" + '/log.json', "w") as f:
            json.dump(episode_stats, f, cls=StableJSONizer)
        with open(result_dir + f"/{idx}/" + '/options.json', "w") as f:
            json.dump(options, f, cls=StableJSONizer)
        if image_dir:
            os.makedirs(image_dir + f"/{idx}", exist_ok=True)
            for img_idx in range(len(images)):
                im = Image.fromarray(images[img_idx])
                im.save(image_dir + f"/{idx}/" + f'{img_idx}.jpg')
