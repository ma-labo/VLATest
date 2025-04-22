"""
Name   : model_interface.py
Author : ZHIJIE WANG
Time   : 7/19/24
"""

import os
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien

from pathlib import Path

SEED = 2024

PACKAGE_DIR = Path(__file__).parent.resolve()

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}

TASKS = [
    "google_robot_pick_customizable",
    "google_robot_pick_customizable_ycb",
    "google_robot_pick_customizable_no_overlay",
    "google_robot_move_near_customizable",
    "google_robot_move_near_customizable_ycb",
    "google_robot_move_near_customizable_no_overlay",
    "widowx_put_on_customizable",
    "widowx_put_on_customizable_ycb",
    "widowx_put_on_customizable_no_overlay",
    "widowx_put_in_customizable",
    "widowx_put_in_customizable_ycb",
    "widowx_put_in_customizable_no_overlay",
]

ckpt_dir = str(PACKAGE_DIR) + '/../checkpoints'

sapien.render_config.rt_use_denoiser = True


class VLAInterface:
    def __init__(self, task, model_name):
        if task in TASKS:
            self.task = task
        else:
            raise ValueError(task)
        if "google" in self.task:
            self.policy_setup = "google_robot"
        else:
            self.policy_setup = "widowx_bridge"
        if "rt_1" in model_name:
            from simpler_env.policies.rt1.rt1_model import RT1Inference

            ckpt_path = os.path.join(ckpt_dir, RT_1_CHECKPOINTS[model_name])
            self.model = RT1Inference(saved_model_path=ckpt_path, policy_setup=self.policy_setup)
        elif "octo" in model_name:
            from simpler_env.policies.octo.octo_model import OctoInference

            self.model = OctoInference(model_type=model_name, policy_setup=self.policy_setup, init_rng=0)
        elif "openvla" in model_name:
            from simpler_env.policies.openvla.openvla_model import OpenVLAInference

            self.model = OpenVLAInference(model_type=model_name, policy_setup=self.policy_setup)
        else:
            raise ValueError(model_name)

    def run_interface(self, seed=None, options=None):
        env = simpler_env.make(self.task)
        obs, reset_info = env.reset(seed=seed, options=options)
        instruction = env.get_language_instruction()
        self.model.reset(instruction)
        print(instruction)
        print("Reset info", reset_info)

        image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
        images = [image]
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        episode_stats = {}
        while not (predicted_terminated or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            raw_action, action = self.model.step(image)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )
            print(timestep, info)
            episode_stats[timestep] = info
            # update image observation
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        print(f"Episode success: {success}")
        env.close()
        del env
        return images, episode_stats


class VLAInterfaceLM(VLAInterface):
    def run_interface(self, seed=None, options=None, instruction=None):
        env = simpler_env.make(self.task)
        obs, reset_info = env.reset(seed=seed, options=options)
        if not instruction:
            instruction = env.get_language_instruction()
        self.model.reset(instruction)
        print(instruction)
        print("Reset info", reset_info)

        image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
        images = [image]
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        episode_stats = {}
        while not (predicted_terminated or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            raw_action, action = self.model.step(image)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )
            print(timestep, info)
            episode_stats[timestep] = info
            # update image observation
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        print(f"Episode success: {success}")
        env.close()
        del env
        return images, episode_stats


if __name__ == '__main__':
    task_name = "google_robot_pick_customizable"
    model = "rt_1_x"  # @param ["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small", "openvla-7b]

    vla = VLAInterface(model_name=model, task=task_name)
