"""
Name   : test_generation.py
Author : ZHIJIE WANG
Time   : 8/7/24
"""

import json
from pathlib import Path
import numpy as np
from transforms3d.euler import euler2quat
import argparse
from tqdm import tqdm
from scipy.spatial import KDTree

# Setup paths
PACKAGE_DIR = Path(__file__).parent.resolve()


class RandomTesting:
    def __init__(self, object_list=None, safe_range=None, seed=None, max_obstacles=None, random_number_obstacles=True):
        if safe_range is None:
            safe_range = [(-0.5, -0.1), (0, 0.4)]
        if seed:
            self.seed = seed
            np.random.seed(seed)
        if max_obstacles >= 0:
            self.max_obstacles = max_obstacles
        else:
            self.max_obstacles = 3
        self.random_number_obstacles = random_number_obstacles
        if object_list:
            if object_list == 'ycb':
                with open(str(PACKAGE_DIR) + '/../ManiSkill2_real2sim/data/ycb-dataset/info_ycb.json', 'r') as f:
                    self.object_list = list(json.load(f).keys())
            else:
                self.object_list = list(json.load(object_list).keys())
        else:
            with open(str(PACKAGE_DIR) + '/../ManiSkill2_real2sim/data/custom/info_pick_custom_v0.json', 'r') as f:
                self.object_list = list(json.load(f).keys())
        self.object_queue = self.object_list.copy()
        self.safe_rage = safe_range

    @staticmethod
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

    def reset(self):
        self.object_queue = self.object_list.copy()

    def query(self):
        obj = np.random.choice(self.object_queue)
        obj_to_remove = [v for v in self.object_queue if
                         self._get_instruction_obj_name(v) == self._get_instruction_obj_name(obj)]
        for candidate in obj_to_remove:
            self.object_queue.remove(candidate)
        pos_x, pos_y = np.random.uniform(*self.safe_rage[0]), np.random.uniform(*self.safe_rage[1])
        orientation = np.random.choice(["standing", "horizontal"])
        if orientation == "horizontal":
            orientation = [v for v in euler2quat(0, 0, np.random.uniform(-np.pi, np.pi))]
        else:
            orientation = [v for v in euler2quat(np.pi / 2, 0, 0)]
        return obj, pos_x, pos_y, orientation

    def to_nl_command(self, obj, pos_x, pos_y, orientation):
        return "ADD %s to (%.4f, %.4f) and the orientation is %s." % (self._get_instruction_obj_name(obj),
                                                                      pos_x, pos_y, [round(v, 4) for v in orientation])

    def generate_nl_commands(self):
        commands = []
        for _ in range(np.random.randint(1,
                                         self.max_obstacles + 2) if self.random_number_obstacles else self.max_obstacles + 1):
            commands.append(self.to_nl_command(*self.query()))
        self.reset()
        return commands


class GraspSingleRandomTesting(RandomTesting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_options(self):
        options = {}
        for i in range(np.random.randint(1,
                                         self.max_obstacles + 2) if self.random_number_obstacles else self.max_obstacles + 1):
            obj, pos_x, pos_y, orientation = self.query()
            if i == 0:
                options["model_id"] = obj
                options["obj_init_options"] = {}
                options["obj_init_options"]['init_xy'] = [pos_x, pos_y]
                options["obj_init_options"]['orientation'] = orientation
            else:
                if "distractor_model_ids" not in options:
                    options["distractor_model_ids"] = [obj]
                    options["distractor_obj_init_options"] = {}
                    options["distractor_obj_init_options"][obj] = {}
                    options["distractor_obj_init_options"][obj]["init_xy"] = [pos_x, pos_y]
                    options["distractor_obj_init_options"][obj]["init_rot_quat"] = orientation
                else:
                    options["distractor_model_ids"].append(obj)
                    options["distractor_obj_init_options"][obj] = {}
                    options["distractor_obj_init_options"][obj]["init_xy"] = [pos_x, pos_y]
                    options["distractor_obj_init_options"][obj]["init_rot_quat"] = orientation
        self.reset()
        return options


class MoveNearRandomTesting(RandomTesting):
    def __init__(self, safe_dist=0.15, **kwargs):
        super().__init__(**kwargs)
        self.omit_space = []
        self.safe_dist = safe_dist

    def _check_safe(self, new_pose):
        for pose in self.omit_space:
            if np.linalg.norm(np.array(pose) - np.array(new_pose)) < self.safe_dist:
                return False
        return True

    def reset(self):
        self.object_queue = self.object_list.copy()
        self.omit_space = []

    def query(self):
        obj = np.random.choice(self.object_queue)
        obj_to_remove = [v for v in self.object_queue if
                         self._get_instruction_obj_name(v) == self._get_instruction_obj_name(obj)]
        for candidate in obj_to_remove:
            self.object_queue.remove(candidate)
        while True:
            pos_x, pos_y = np.random.uniform(*self.safe_rage[0]), np.random.uniform(*self.safe_rage[1])
            if self._check_safe([pos_x, pos_y]):
                break
        self.omit_space.append([pos_x, pos_y])
        orientation = np.random.choice(["standing", "horizontal"])
        if orientation == "horizontal":
            orientation = [v for v in euler2quat(0, 0, np.random.uniform(-np.pi, np.pi))]
        else:
            orientation = [v for v in euler2quat(np.pi / 2, 0, 0)]
        return obj, pos_x, pos_y, orientation

    def generate_options(self):
        options = {}
        for i in range(np.random.randint(2,
                                         self.max_obstacles + 3) if self.random_number_obstacles else self.max_obstacles + 2):
            obj, pos_x, pos_y, orientation = self.query()
            if "model_ids" not in options:
                options["model_ids"] = [obj]
                options["obj_init_options"] = {}
                options["obj_init_options"][obj] = {}
                options["obj_init_options"][obj]["init_xy"] = [pos_x, pos_y]
                options["obj_init_options"][obj]["init_rot_quat"] = orientation
            else:
                options["model_ids"].append(obj)
                options["obj_init_options"][obj] = {}
                options["obj_init_options"][obj]["init_xy"] = [pos_x, pos_y]
                options["obj_init_options"][obj]["init_rot_quat"] = orientation
        obj_indexes = [v for v in range(len(options["model_ids"]))]
        np.random.shuffle(obj_indexes)
        options["source_obj_id"] = int(obj_indexes[0])
        options["target_obj_id"] = int(obj_indexes[1])
        self.reset()
        return options


class PutOnRandomTesting(MoveNearRandomTesting):
    def __init__(self, safe_dist=0.05, safe_range=None, target_list=None, **kwargs):
        if not safe_range:
            safe_range = [(-0.3, 0), (-0.16, 0.12)]
        if target_list:
            self.target_list = target_list
        else:
            self.target_list = ["bridge_plate_objaverse", "bridge_plate_objaverse_larger", "table_cloth_generated", "table_cloth_generated_shorter"]

        super().__init__(safe_dist=safe_dist, safe_range=safe_range, **kwargs)

    def generate_options(self):
        options = {}

        # source obj
        obj, pos_x, pos_y, orientation = self.query()
        options["model_ids"] = [obj]
        options["obj_init_options"] = {}
        options["obj_init_options"][obj] = {}
        options["obj_init_options"][obj]["init_xy"] = [pos_x, pos_y]
        options["obj_init_options"][obj]["init_rot_quat"] = orientation

        # target obj
        while True:
            pos_x, pos_y = np.random.uniform(*self.safe_rage[0]), np.random.uniform(*self.safe_rage[1])
            if self._check_safe([pos_x, pos_y]):
                break
        self.omit_space.append([pos_x, pos_y])
        obj = np.random.choice(self.target_list)
        options["model_ids"].append(obj)
        options["obj_init_options"][obj] = {}
        options["obj_init_options"][obj]["init_xy"] = [pos_x, pos_y]
        options["obj_init_options"][obj]["init_rot_quat"] = [1, 0, 0, 0]

        for i in range(np.random.randint(0, self.max_obstacles + 1) if self.random_number_obstacles else self.max_obstacles):
            obj, pos_x, pos_y, orientation = self.query()
            options["model_ids"].append(obj)
            options["obj_init_options"][obj] = {}
            options["obj_init_options"][obj]["init_xy"] = [pos_x, pos_y]
            options["obj_init_options"][obj]["init_rot_quat"] = orientation
        options["source_obj_id"] = 0
        options["target_obj_id"] = 1
        self.reset()
        return options


class PutInRandomTesting(PutOnRandomTesting):
    def __init__(self, safe_dist=0, safe_range=None, **kwargs):
        if not safe_range:
            safe_range = [(-0.2, -0.1), (0.1, 0.2)]

        super().__init__(safe_dist=safe_dist, safe_range=safe_range, **kwargs)

    def generate_options(self):
        options = {}

        # source obj
        obj, pos_x, pos_y, orientation = self.query()
        options["model_ids"] = [obj]
        options["obj_init_options"] = {}
        options["obj_init_options"][obj] = {}
        options["obj_init_options"][obj]["init_xy"] = [pos_x, pos_y]
        options["obj_init_options"][obj]["init_rot_quat"] = orientation

        # target obj
        obj = "dummy_sink_target_plane"
        options["model_ids"].append("dummy_sink_target_plane")
        options["obj_init_options"][obj] = {}
        options["obj_init_options"][obj]["init_xy"] = [-0.125, 0.025]
        options["obj_init_options"][obj]["init_rot_quat"] = [1, 0, 0, 0]

        for i in range(np.random.randint(0, self.max_obstacles + 1) if self.random_number_obstacles else self.max_obstacles):
            obj, pos_x, pos_y, orientation = self.query()
            options["model_ids"].append(obj)
            options["obj_init_options"][obj] = {}
            options["obj_init_options"][obj]["init_xy"] = [pos_x, pos_y]
            options["obj_init_options"][obj]["init_rot_quat"] = orientation
        options["source_obj_id"] = 0
        options["target_obj_id"] = 1
        self.reset()
        return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="VLA Fuzzing")
    parser.add_argument('-t', '--task', type=str, choices=["grasp", "move", "put-in", "put-on"], default="grasp", help="VLA Task")
    parser.add_argument('-n', '--num', type=int, default=100, help="Number of scenarios to generate")
    parser.add_argument('--nl', type=bool, default=False, help="Return natural language commands")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random Seed")
    parser.add_argument('--ro', type=bool, default=True, help="Random number of obstacles")
    parser.add_argument('--obstacles', type=int, default=3, help="Max number of obstacles")
    parser.add_argument('-o', '--output', type=str, help="Output path, e.g., folder")
    parser.add_argument('--ycb', type=bool, default=False, help="Use YCB dataset")

    args = parser.parse_args()

    random_seed = args.seed if args.seed else np.random.randint(0, 4294967295)  # max uint32

    if args.task == "grasp":
        fuzzer = GraspSingleRandomTesting(seed=random_seed, max_obstacles=args.obstacles,
                                          random_number_obstacles=args.ro, object_list="ycb" if args.ycb else None)
    elif args.task == "move":
        fuzzer = MoveNearRandomTesting(seed=random_seed, max_obstacles=args.obstacles,
                                       random_number_obstacles=args.ro, object_list="ycb" if args.ycb else None)
    elif args.task == "put-on":
        fuzzer = PutOnRandomTesting(seed=random_seed, max_obstacles=args.obstacles,
                                    random_number_obstacles=args.ro, object_list="ycb" if args.ycb else None)
    elif args.task == "put-in":
        fuzzer = PutInRandomTesting(seed=random_seed, max_obstacles=args.obstacles,
                                    random_number_obstacles=args.ro, object_list="ycb" if args.ycb else None)
    else:
        raise NotImplementedError

    output_name = ""

    if args.ycb:
        output_name += 'ycb_'

    res = {}
    for i in tqdm(range(args.num)):
        if args.nl:
            res[i] = fuzzer.generate_nl_commands()
        else:
            res[i] = fuzzer.generate_options()

    res["seed"] = random_seed

    res["num"] = args.num

    if args.ro:
        output_name += f"t-{args.task}_n-{args.num}_o-m{args.obstacles}_s-{random_seed}.json"
    else:
        output_name += f"t-{args.task}_n-{args.num}_o-{args.obstacles}_s-{random_seed}.json"

    output_path = args.output + output_name if args.output else str(PACKAGE_DIR) + "/../data/" + output_name

    with open(output_path, 'w') as f:
        json.dump(res, f)
