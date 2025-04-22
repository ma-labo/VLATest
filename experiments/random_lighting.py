"""
Name   : random_lighting.py
Author : ZHIJIE WANG
Time   : 8/15/24
"""

import json
from pathlib import Path
import numpy as np
from transforms3d.euler import euler2quat
import argparse
from tqdm import tqdm

# Setup paths
PACKAGE_DIR = Path(__file__).parent.resolve()


class RandomLighting:
    def __init__(self, direction=None, seed=None, factor_range=None, step_range=None):
        if seed:
            self.seed = seed
            np.random.seed(seed)
        self.factor_range = factor_range if factor_range else (1, 2)
        self.step_range = step_range if step_range else (0, 5)
        self.direction = direction

    def query(self):
        # direction, step, factor = self.lighting_cfgs[0], self.lighting_cfgs[1], self.lighting_cfgs[2]
        if self.direction:
            direction = self.direction
        else:
            direction = np.random.choice(["BRIGHT", "DARK"])
        if direction == 'DARK':
            factor = np.random.uniform(1 / self.factor_range[1], 1 / self.factor_range[0])
            factor = 1 / factor
        else:
            factor = np.random.uniform(*self.factor_range)
        step = np.random.randint(*self.step_range)
        return direction, step, factor

    def generate_options(self):
        direction, step, factor = self.query()
        if step == 0:
            return {"lighting_cfgs": "DEFAULT"}
        return {"lighting_cfgs": [direction, int(step), float(factor)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="VLA Fuzzing")
    parser.add_argument('-n', '--num', type=int, default=100, help="Number of scenarios to generate")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random Seed")
    parser.add_argument('-o', '--output', type=str, help="Output path, e.g., folder")
    parser.add_argument('-d', '--direction', type=str, default=None, help="Lighting direction")

    args = parser.parse_args()

    random_seed = args.seed if args.seed else np.random.randint(0, 4294967295)  # max uint32

    fuzzer = RandomLighting(args.direction, random_seed)

    output_name = ""

    res = {}
    for i in tqdm(range(args.num)):
        res[i] = fuzzer.generate_options()

    res["seed"] = random_seed

    res["num"] = args.num

    if args.direction:
        output_name += f"lighting_n-{args.num}_d-{args.direction}_s-{random_seed}.json"
    else:
        output_name += f"lighting_n-{args.num}_s-{random_seed}.json"

    output_path = args.output + output_name if args.output else str(PACKAGE_DIR) + "/../data/" + output_name

    with open(output_path, 'w') as f:
        json.dump(res, f)
