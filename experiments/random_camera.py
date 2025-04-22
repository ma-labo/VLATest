"""
Name   : random_camera.py
Author : ZHIJIE WANG
Time   : 8/16/24
"""

import json
from pathlib import Path
import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat, mat2quat
import argparse
from tqdm import tqdm

# Setup paths
PACKAGE_DIR = Path(__file__).parent.resolve()


class RandomCamera:
    def __init__(self, base=None, seed=None):
        if seed:
            self.seed = seed
            np.random.seed(seed)
        if base == 'google':
            self.camera = 'overhead_camera'
            self.base_pos = [0, 0, 0]
            self.base_rot = [0.5, 0.5, -0.5, 0.5]
        elif base == 'widowx':
            self.camera = "3rd_view_camera"
            self.base_pos = [0.0, -0.16, 0.36]
            self.base_rot = [0.89929167, -0.09263245, 0.35892477, 0.23209206]
        elif base == 'widowx_sink':
            self.camera = "3rd_view_camera"
            self.base_pos = [-0.00300001, -0.21, 0.39]
            self.base_rot = [-0.907313, 0.0782, -0.36434, -0.194741]
        else:
            raise NotImplementedError

    def query(self):
        pos = [np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)]
        rot_y_plane = np.random.uniform(-np.pi/36, np.pi/36)
        rot_z_plane = np.random.uniform(-np.pi/36, np.pi/36)
        r = quat2mat(self.base_rot)
        new_r = r @ quat2mat(euler2quat(0, rot_y_plane, rot_z_plane))
        new_rot = mat2quat(new_r)

        return [pos[0] + self.base_pos[0], pos[1] + self.base_pos[1], pos[2] + self.base_pos[2]], new_rot.tolist()

    def generate_options(self):
        p, q = self.query()
        options = {
            "camera_cfgs": {
                self.camera: {
                    "p": p,
                    "q": q
                }
            }
        }
        return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="VLA Fuzzing")
    parser.add_argument('-n', '--num', type=int, default=1000, help="Number of scenarios to generate")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random Seed")
    parser.add_argument('-o', '--output', type=str, help="Output path, e.g., folder")
    parser.add_argument('-b', '--base', type=str, default="google", help="Camera base")

    args = parser.parse_args()

    random_seed = args.seed if args.seed else np.random.randint(0, 4294967295)  # max uint32

    fuzzer = RandomCamera(args.base, random_seed)

    output_name = ""

    res = {}
    for i in tqdm(range(args.num)):
        res[i] = fuzzer.generate_options()

    res["seed"] = random_seed

    res["num"] = args.num

    output_name += f"camera_n-{args.num}_b-{args.base}_s-{random_seed}.json"

    output_path = args.output + output_name if args.output else str(PACKAGE_DIR) + "/../data/" + output_name

    with open(output_path, 'w') as f:
        json.dump(res, f)
