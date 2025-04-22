"""
Name   : run_fuzzer_w_camera.py
Author : ZHIJIE WANG
Time   : 8/31/24
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
from experiments.random_camera import RandomCamera

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
    parser.add_argument('-m', '--model', type=str,
                        choices=["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small", "openvla-7b"],
                        default="rt_1_x",
                        help="VLA model")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random Seed")
    parser.add_argument('-r', '--resume', type=bool, default=True, help="Resume from where we left.")
    parser.add_argument('-t', '--task', type=str, choices=["grasp", "move", "put-on", "put-in"],
                        default='grasp', help="Tasks.")
    parser.add_argument('-sa', '--samples', type=int, default=3, help="samples.")

    args = parser.parse_args()

    dataset_path = args.data if args.data else str(PACKAGE_DIR) + "/../data/all-correct-task.json"

    camera_random_seed = args.seed if args.seed else np.random.randint(0, 4294967295)  # max uint32

    if args.task == 'grasp' or args.task == 'move':
        camera_fuzzer = RandomCamera(base='google', seed=camera_random_seed)
    elif args.task == 'put-on':
        camera_fuzzer = RandomCamera(base='widowx', seed=camera_random_seed)
    elif args.task == 'put-in':
        camera_fuzzer = RandomCamera(base='widowx_sink', seed=camera_random_seed)
    else:
        raise NotImplementedError

    if args.output:
        result_dir = args.output + f"random_camera_{camera_random_seed}"
    else:
        result_dir = str(PACKAGE_DIR) + "/../results/" + f"random_camera_{camera_random_seed}"

    os.makedirs(result_dir, exist_ok=True)

    if args.image_output:
        image_dir = args.image_output + f"random_camera_{camera_random_seed}"
        os.makedirs(image_dir, exist_ok=True)
    else:
        image_dir = None

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    camera = {}

    for dataset_name in dataset.keys():

        camera[dataset_name] = {}

        data_path = str(PACKAGE_DIR) + f"/../data/{dataset_name}.json"

        for model_name in dataset[dataset_name].keys():

            camera[dataset_name][model_name] = {}

            random_seed = int(model_name.split('_')[-1])

            model = model_name[:-len('_' + str(random_seed))]

            valid_tasks = dataset[dataset_name][model_name]

            with open(data_path, 'r') as f:
                tasks = json.load(f)

            for idx in range(tasks["num"]):
                if idx in valid_tasks:
                    for sample in range(args.samples):
                        camera[dataset_name][model_name][f"{idx}_{sample}"] = camera_fuzzer.generate_options()

    for dataset_name in dataset.keys():

        if args.task in dataset_name:

            data_path = str(PACKAGE_DIR) + f"/../data/{dataset_name}.json"

            for model_name in dataset[dataset_name].keys():

                random_seed = int(model_name.split('_')[-1])

                model = model_name[:-len('_' + str(random_seed))]

                if model == args.model:
                    valid_tasks = dataset[dataset_name][model_name]

                    if "grasp" in dataset_name:
                        vla = VLAInterface(model_name=args.model, task="google_robot_pick_customizable_no_overlay")
                    elif "move" in dataset_name:
                        vla = VLAInterface(model_name=args.model, task="google_robot_move_near_customizable_no_overlay")
                    elif "put-on" in dataset_name:
                        vla = VLAInterface(model_name=args.model, task="widowx_put_on_customizable_no_overlay")
                    elif "put-in" in dataset_name:
                        vla = VLAInterface(model_name=args.model, task="widowx_put_in_customizable_no_overlay")
                    else:
                        raise NotImplementedError

                    with open(data_path, 'r') as f:
                        tasks = json.load(f)

                    for idx in tqdm(valid_tasks):
                        for sample in range(args.samples):
                            camera_options = camera[dataset_name][model_name][f"{idx}_{sample}"]
                            print(camera_options)
                            options = tasks[str(idx)]
                            options.update(camera_options)

                            if args.resume and os.path.exists(result_dir + f"/{dataset_name}_{model_name}_{idx}_{sample}/" + '/log.json'):  # if resume allowed then skip the finished runs.
                                continue
                            images, episode_stats = vla.run_interface(seed=random_seed, options=options)
                            os.makedirs(result_dir + f"/{dataset_name}_{model_name}_{idx}_{sample}", exist_ok=True)
                            with open(result_dir + f"/{dataset_name}_{model_name}_{idx}_{sample}/" + '/log.json', "w") as f:
                                json.dump(episode_stats, f, cls=StableJSONizer)
                            if image_dir:
                                os.makedirs(image_dir + f"/{dataset_name}_{model_name}_{idx}_{sample}", exist_ok=True)
                                for img_idx in range(len(images)):
                                    im = Image.fromarray(images[img_idx])
                                    im.save(image_dir + f"/{dataset_name}_{model_name}_{idx}_{sample}/" + f'{img_idx}.jpg')
                            with open(result_dir + f"/{dataset_name}_{model_name}_{idx}_{sample}/" + '/options.json', "w") as f:
                                json.dump(options, f, cls=StableJSONizer)
