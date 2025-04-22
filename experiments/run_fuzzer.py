"""
Name   : run_fuzzer.py
Author : ZHIJIE WANG
Time   : 8/8/24
"""
import argparse
import numpy as np
from experiments.model_interface import VLAInterface
from pathlib import Path
from tqdm import tqdm
import json
import os
from PIL import Image
import shutil

# Setup paths
PACKAGE_DIR = Path(__file__).parent.resolve()


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

    if "grasp" in dataset_name:
        if 'ycb' in dataset_name:
            vla = VLAInterface(model_name=args.model, task="google_robot_pick_customizable_ycb")
        else:
            vla = VLAInterface(model_name=args.model, task="google_robot_pick_customizable")
    elif "move" in dataset_name:
        if 'ycb' in dataset_name:
            vla = VLAInterface(model_name=args.model, task="google_robot_move_near_customizable_ycb")
        else:
            vla = VLAInterface(model_name=args.model, task="google_robot_move_near_customizable")
    elif "put-on" in dataset_name:
        if 'ycb' in dataset_name:
            vla = VLAInterface(model_name=args.model, task="widowx_put_on_customizable_ycb")
        else:
            vla = VLAInterface(model_name=args.model, task="widowx_put_on_customizable")
    elif "put-in" in dataset_name:
        if 'ycb' in dataset_name:
            vla = VLAInterface(model_name=args.model, task="widowx_put_in_customizable_ycb")
        else:
            vla = VLAInterface(model_name=args.model, task="widowx_put_in_customizable")
    else:
        raise NotImplementedError

    with open(data_path, 'r') as f:
        tasks = json.load(f)

    if args.output:
        result_dir = args.output + data_path.split('/')[-1].split(".")[0]
    else:
        result_dir = str(PACKAGE_DIR) + "/../results/" + data_path.split('/')[-1].split(".")[0]
    os.makedirs(result_dir, exist_ok=True)
    result_dir += f'/{args.model}_{random_seed}'
    if not args.resume:
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    if args.image_output:
        image_dir = args.image_output + data_path.split('/')[-1].split(".")[0]
        os.makedirs(image_dir, exist_ok=True)
        image_dir += f'/{args.model}_{random_seed}'
        os.makedirs(image_dir, exist_ok=True)
    else:
        image_dir = None

    for idx in tqdm(range(tasks["num"])):
        if args.resume and os.path.exists(result_dir + f"/{idx}/" + '/log.json'):  # if resume allowed then skip the finished runs.
            continue
        options = tasks[str(idx)]
        images, episode_stats = vla.run_interface(seed=random_seed, options=options)
        os.makedirs(result_dir + f"/{idx}", exist_ok=True)
        with open(result_dir + f"/{idx}/" + '/log.json', "w") as f:
            json.dump(episode_stats, f, cls=StableJSONizer)
        if image_dir:
            os.makedirs(image_dir + f"/{idx}", exist_ok=True)
            for img_idx in range(len(images)):
                im = Image.fromarray(images[img_idx])
                im.save(image_dir + f"/{idx}/" + f'{img_idx}.jpg')