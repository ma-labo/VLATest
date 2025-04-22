from collections import OrderedDict
from typing import List, Optional

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose

from .base_env import CustomSceneEnv, CustomOtherObjectsInSceneEnv


class GraspSingleInSceneCustomizableEnv(CustomSceneEnv):
    obj: sapien.Actor  # target object to grasp

    def __init__(
            self,
            require_lifting_obj_for_success: bool = True,
            success_from_episode_stats: bool = True,
            distractor_model_ids: Optional[List[str]] = None,
            slightly_darker_lighting: bool = False,
            slightly_brighter_lighting: bool = False,
            darker_lighting: bool = False,
            prepackaged_config: bool = False,
            **kwargs,
    ):
        if isinstance(distractor_model_ids, str):
            distractor_model_ids = [distractor_model_ids]
        self.distractor_model_ids = distractor_model_ids

        self.model_id = None
        self.model_scale = None
        self.model_bbox_size = None

        self.selected_distractor_model_ids = None
        self.selected_distractor_model_scales = None

        self.obj = None
        self.distractor_objs = []

        self.obj_init_options = {}
        self.distractor_obj_init_options = {}

        self.slightly_darker_lighting = slightly_darker_lighting
        self.slightly_brighter_lighting = slightly_brighter_lighting
        self.darker_lighting = darker_lighting

        self.require_lifting_obj_for_success = require_lifting_obj_for_success
        self.success_from_episode_stats = success_from_episode_stats
        self.consecutive_grasp = 0
        self.lifted_obj = False
        self.obj_height_after_settle = None
        self.episode_stats = None

        self.lighting_cfgs = None

        self.prepackaged_config = prepackaged_config
        if self.prepackaged_config:
            # use prepackaged evaluation configs (visual matching)
            kwargs.update(self._setup_prepackaged_env_init_config())

        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret["control_mode"] = (
            "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        )
        ret["scene_name"] = "google_pick_coke_can_1_v4"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/google_coke_can_real_eval_1.png"
        )
        ret["rgb_overlay_cameras"] = ["overhead_camera"]

        return ret

    def _load_actors(self):
        self._load_arena_helper()
        self._load_model()
        self.obj.set_damping(0.1, 0.1)

    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        # remove distractor objects
        for distractor_obj in self.distractor_objs:
            self._scene.remove_actor(distractor_obj)
        self.distractor_objs = []

        if options is None:
            options = dict()
        options = options.copy()

        self.obj_init_options = options.get("obj_init_options", {})
        self.distractor_obj_init_options = options.get(
            "distractor_obj_init_options", {}
        )

        # set objects and distractor objects
        self.set_episode_rng(seed)
        model_scale = options.get("model_scale", None)
        model_id = options.get("model_id", None)
        reconfigure = options.get("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        # if self.distractor_model_ids is not None:
        distractor_model_scales = options.get("distractor_model_scales", None)
        distractor_model_ids = options.get("distractor_model_ids", None)
        if distractor_model_ids is not None:
            reconfigure = True
            self._set_distractor_models(
                distractor_model_ids, distractor_model_scales
            )

        if self.prepackaged_config:
            _reconfigure = self._additional_prepackaged_config_reset(options)
            reconfigure = reconfigure or _reconfigure

        if options.get("camera_cfgs", None) is not None:
            reconfigure = True

        lighting_cfgs = options.get("lighting_cfgs", None)

        if lighting_cfgs is not None:
            self.lighting_cfgs = lighting_cfgs
            reconfigure = True

        options["reconfigure"] = reconfigure

        self.consecutive_grasp = 0
        self.lifted_obj = False
        self.obj_height_after_settle = None
        # episode-level info
        self._initialize_episode_stats()

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update(
            {
                "model_id": self.model_id,
                "model_scale": self.model_scale,
                "distractor_model_ids": self.selected_distractor_model_ids,
                "distractor_model_scales": self.selected_distractor_model_scales,
                "obj_init_pose_wrt_robot_base": self.agent.robot.pose.inv()
                                                * self.obj.pose,
            }
        )
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.35, 0.20],
            "init_rot_quat": [0, 0, 0, 1],
        }
        new_urdf_version = self._episode_rng.choice(
            [
                "",
                "recolor_tabletop_visual_matching_1",
                "recolor_tabletop_visual_matching_2",
                "recolor_cabinet_visual_matching_1",
            ]
        )
        if new_urdf_version != self.urdf_version:
            self.urdf_version = new_urdf_version
            self._configure_agent()
            return True
        return False

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            n_lift_significant=0,
            consec_grasp=False,
            grasped=False,
        )

    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        if isinstance(self.lighting_cfgs, list):
            direction, step, factor = self.lighting_cfgs[0], self.lighting_cfgs[1], self.lighting_cfgs[2]
            if direction == 'DARK':
                factor = 1 / factor
            # default lighting
            self._scene.add_directional_light(
                [0, 0, -1],
                (np.array([2.2, 2.2, 2.2]) * (factor ** step)).tolist(),
                shadow=shadow,
                scale=5,
                shadow_map_size=2048,
            )
            self._scene.add_directional_light([-1, -0.5, -1], (np.array([0.7, 0.7, 0.7]) * (factor ** step)).tolist())
            self._scene.add_directional_light([1, 1, -1], (np.array([0.7, 0.7, 0.7]) * (factor ** step)).tolist())
        else:
            if isinstance(self.lighting_cfgs, str):
                if self.lighting_cfgs == "slightly_brighter":
                    self.slightly_brighter_lighting = True
                    self.slightly_darker_lighting = False
                    self.darker_lighting = False
                elif self.lighting_cfgs == "slightly_darker":
                    self.slightly_brighter_lighting = False
                    self.slightly_darker_lighting = True
                    self.darker_lighting = False
                elif self.lighting_cfgs == "darker":
                    self.slightly_brighter_lighting = False
                    self.slightly_darker_lighting = False
                    self.darker_lighting = True
                else:
                    self.slightly_brighter_lighting = False
                    self.slightly_darker_lighting = False
                    self.darker_lighting = False
            if self.slightly_brighter_lighting:
                self._scene.add_directional_light(
                    [0, 0, -1],
                    [3.6, 3.6, 3.6],
                    shadow=shadow,
                    scale=5,
                    shadow_map_size=2048,
                )
                self._scene.add_directional_light([-1, -0.5, -1], [1.3, 1.3, 1.3])
                self._scene.add_directional_light([1, 1, -1], [1.3, 1.3, 1.3])
            elif self.slightly_darker_lighting:
                self._scene.add_directional_light(
                    [1, 1, -1],
                    [0.8, 0.8, 0.8],
                    shadow=shadow,
                    scale=5,
                    shadow_map_size=2048,
                )
                self._scene.add_directional_light([0, 0, -1], [0.8, 0.8, 0.8])
            elif self.darker_lighting:
                self._scene.add_directional_light(
                    [1, 1, -1],
                    [0.3, 0.3, 0.3],
                    shadow=shadow,
                    scale=5,
                    shadow_map_size=2048,
                )
                self._scene.add_directional_light([0, 0, -1], [0.3, 0.3, 0.3])
            else:
                # default lighting
                self._scene.add_directional_light(
                    [0, 0, -1],
                    [2.2, 2.2, 2.2],
                    shadow=shadow,
                    scale=5,
                    shadow_map_size=2048,
                )
                self._scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
                self._scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly from self.model_ids."""
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        if model_scale is None:
            model_scales = self.model_db[self.model_id].get("scales")
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure

    def _set_distractor_models(self, distractor_model_ids, distractor_model_scales):
        assert distractor_model_ids is not None

        self.selected_distractor_model_ids = distractor_model_ids

        if distractor_model_scales is None:
            distractor_model_scales = []
            for distractor_model_id in distractor_model_ids:
                model_scales = self.model_db[distractor_model_id].get("scales")
                if model_scales is None:
                    model_scale = 1.0
                else:
                    model_scale = random_choice(model_scales, self._episode_rng)
                distractor_model_scales.append(model_scale)

        self.selected_distractor_model_scales = distractor_model_scales

    def _initialize_actors(self):
        # The object will fall from a certain initial height
        obj_init_xy = self.obj_init_options.get("init_xy", None)
        if obj_init_xy is None:
            obj_init_xy = self._episode_rng.uniform([-0.35, -0.02], [-0.12, 0.42], [2])
        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z + 0.5  # let object fall onto the table
        obj_init_rot_quat = self.obj_init_options.get("init_rot_quat", [1, 0, 0, 0])
        p = np.hstack([obj_init_xy, obj_init_z])
        q = obj_init_rot_quat

        # Rotate along z-axis
        if self.obj_init_options.get("init_rand_rot_z", False):
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = qmult(euler2quat(0, 0, ori), q)

        # Rotate along a random axis by a small angle
        if (
                init_rand_axis_rot_range := self.obj_init_options.get(
                    "init_rand_axis_rot_range", 0.0
                )
        ) > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, init_rand_axis_rot_range)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.obj.set_pose(sapien.Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later in _initialize_agent (in base_env.py)
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # Lock rotation around x and y to let the target object fall onto the table
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE: Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        # Record the object height after it settles
        self.obj_height_after_settle = self.obj.pose.p[2]

        if len(self.distractor_objs) > 0:
            # Set distractor objects
            for distractor_obj in self.distractor_objs:
                distractor_obj_init_options = self.distractor_obj_init_options.get(
                    distractor_obj.name, {}
                )

                distractor_init_xy = distractor_obj_init_options.get("init_xy", None)
                if distractor_init_xy is None:
                    distractor_offset_xy = distractor_obj_init_options.get("offset_xy", None)
                    if distractor_offset_xy:
                        distractor_init_xy = obj_init_xy + np.array(distractor_offset_xy)
                    else:
                        while True:
                            distractor_init_xy = obj_init_xy + self._episode_rng.uniform(
                                -0.3, 0.3, [2]
                            )  # hardcoded for now
                            distractor_init_xy = np.clip(
                                distractor_init_xy, [-0.50, 0.05], [-0.1, 0.35]
                            )
                            if np.linalg.norm(distractor_init_xy - obj_init_xy) > 0.25:
                                break
                p = np.hstack(
                    [distractor_init_xy, obj_init_z]
                )  # let distractor fall from the same height as the main object

                distractor_init_rot_quat = distractor_obj_init_options.get(
                    "init_rot_quat", None
                )
                if not distractor_init_rot_quat:
                    orientation = distractor_obj_init_options.get("orientation", None)
                    if orientation:
                        if "standing" in orientation:
                            distractor_init_rot_quat = self.orientations_dict[
                                "upright"
                            ]
                        elif "horizontal" in orientation:
                            distractor_init_rot_quat = self.orientations_dict[
                                "lr_switch"
                            ]
                        else:
                            distractor_init_rot_quat = orientation
                q = (
                    obj_init_rot_quat
                    if distractor_init_rot_quat is None
                    else distractor_init_rot_quat
                )
                distractor_obj.set_pose(sapien.Pose(p, q))
                distractor_obj.set_velocity(np.zeros(3))
                distractor_obj.set_angular_velocity(np.zeros(3))
                # Lock rotation around x and y
                distractor_obj.lock_motion(1, 1, 0, 1, 1, 0)

                # debug
                # sim_steps = int(self.sim_freq * 0.5)
                # for _ in range(sim_steps):
                #     print(distractor_obj.pose)
                #     while True:
                #         self.render_human()
                #         sapien_viewer = self.viewer
                #         if sapien_viewer.window.key_down("0"):
                #             break
                #     self._scene.step()

                # Let distractor objects fall onto the table
                self._settle(0.5)

            # Unlock motion
            for distractor_obj in self.distractor_objs:
                distractor_obj.lock_motion(0, 0, 0, 0, 0, 0)
                distractor_obj.set_pose(distractor_obj.pose)
                distractor_obj.set_velocity(np.zeros(3))
                distractor_obj.set_angular_velocity(np.zeros(3))
                self._settle(0.5)

            lin_vel, ang_vel = 0.0, 0.0
            for distractor_obj in self.distractor_objs:
                lin_vel += np.linalg.norm(distractor_obj.velocity)
                ang_vel += np.linalg.norm(distractor_obj.angular_velocity)
            if lin_vel > 1e-3 or ang_vel > 1e-2:
                self._settle(1.5)

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.obj.pose.transform(self.obj.cmass_local_pose)

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj_pose),
                tcp_to_obj_pos=self.obj_pose.p - self.tcp.pose.p,
            )
        return obs

    def evaluate(self, **kwargs):
        # evaluate the success of the task

        is_grasped = self.agent.check_grasp(self.obj, max_angle=80)
        if is_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
            self.lifted_obj = False

        contacts = self._scene.get_contacts()
        flag = True
        robot_link_names = [x.name for x in self.agent.robot.get_links()]
        for contact in contacts:
            actor_0, actor_1 = contact.actor0, contact.actor1
            other_obj_contact_actor_name = None
            if actor_0.name == self.obj.name:
                other_obj_contact_actor_name = actor_1.name
            elif actor_1.name == self.obj.name:
                other_obj_contact_actor_name = actor_0.name
            if other_obj_contact_actor_name is not None:
                # the object is in contact with an actor
                contact_impulse = np.sum(
                    [point.impulse for point in contact.points], axis=0
                )
                if (other_obj_contact_actor_name not in robot_link_names) and (
                        np.linalg.norm(contact_impulse) > 1e-6
                ):
                    # the object has contact with an actor other than the robot link, so the object is not yet lifted up
                    # print(other_obj_contact_actor_name, np.linalg.norm(contact_impulse))
                    flag = False
                    break

        consecutive_grasp = self.consecutive_grasp >= 5
        diff_obj_height = self.obj.pose.p[2] - self.obj_height_after_settle
        self.lifted_obj = self.lifted_obj or (flag and (diff_obj_height > 0.01))
        lifted_object_significantly = self.lifted_obj and (diff_obj_height > 0.02)

        if self.require_lifting_obj_for_success:
            success = self.lifted_obj
        else:
            success = consecutive_grasp

        self.episode_stats["n_lift_significant"] += int(lifted_object_significantly)
        self.episode_stats["consec_grasp"] = (
                self.episode_stats["consec_grasp"] or consecutive_grasp
        )
        self.episode_stats["grasped"] = self.episode_stats["grasped"] or is_grasped
        if self.success_from_episode_stats:
            # During evaluation, if policy puts down coke can in the end but has lifted it significantly before, it is still a success
            # However, if you want to perform RL training on this environment, make sure to turn off this option
            success = success or (self.episode_stats["n_lift_significant"] >= 5)

        return dict(
            is_grasped=is_grasped,
            consecutive_grasp=consecutive_grasp,
            lifted_object=self.lifted_obj,
            lifted_object_significantly=lifted_object_significantly,
            success=success,
            episode_stats=self.episode_stats,
        )


# ---------------------------------------------------------------------------- #
# Custom Assets
# ---------------------------------------------------------------------------- #


@register_env("GraspSingleCustomizable-v0", max_episode_steps=80)
class GraspSingleCustomInSceneEnv(GraspSingleInSceneCustomizableEnv, CustomOtherObjectsInSceneEnv):
    def __init__(
            self,
            upright: bool = False,
            laid_vertically: bool = False,
            lr_switch: bool = False,
            **kwargs,
    ):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["apple"]
        if upright:
            self.orientation = "upright"
        elif laid_vertically:
            self.orientation = "laid_vertically"
        elif lr_switch:
            self.orientation = "lr_switch"
        else:
            self.orientation = None
        self.orientations_dict = {
            "upright": euler2quat(np.pi / 2, 0, 0),
            "laid_vertically": euler2quat(0, 0, np.pi / 2),
            "lr_switch": euler2quat(0, 0, np.pi),
        }
        super().__init__(**kwargs)

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)

        self.obj = self._build_actor_helper(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id

        if self.selected_distractor_model_ids is not None:
            for distractor_model_id, distractor_model_scale in zip(
                    self.selected_distractor_model_ids,
                    self.selected_distractor_model_scales,
            ):
                distractor_obj = self._build_actor_helper(
                    distractor_model_id,
                    self._scene,
                    scale=distractor_model_scale,
                    density=self.model_db[distractor_model_id].get("density", 1000),
                    physical_material=self._scene.create_physical_material(
                        static_friction=self.obj_static_friction,
                        dynamic_friction=self.obj_dynamic_friction,
                        restitution=0.0,
                    ),
                    root_dir=self.asset_root,
                )
                distractor_obj.name = distractor_model_id
                self.distractor_objs.append(distractor_obj)

    def _get_init_z(self):
        bbox_min = self.model_db[self.model_id]["bbox"]["min"]
        return -bbox_min[2] * self.model_scale + 0.05

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.obj.name)
        task_description = f"pick {obj_name}"
        return task_description

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", None)
        if obj_init_options is None:
            obj_init_options = dict()
        obj_init_options = (
            obj_init_options.copy()
        )  # avoid modifying the original options

        orientation = None
        if obj_init_options.get("init_rot_quat", None) is None:
            if obj_init_options.get("orientation", None) is not None:
                orientation = obj_init_options["orientation"]
            else:
                orientation = self.orientation
            if orientation is not None:
                if type(orientation) == str:
                    try:
                        obj_init_options["init_rot_quat"] = self.orientations_dict[
                            orientation
                        ]
                    except KeyError as e:
                        if "standing" in orientation:
                            obj_init_options["init_rot_quat"] = self.orientations_dict[
                                "upright"
                            ]
                        elif "horizontal" in orientation:
                            obj_init_options["init_rot_quat"] = self.orientations_dict[
                                "lr_switch"
                            ]
                        else:
                            raise e
                else:
                    obj_init_options["init_rot_quat"] = orientation
            else:
                orientation = self._episode_rng.choice(
                    list(self.orientations_dict.keys())
                )
                obj_init_options["init_rot_quat"] = self.orientations_dict[orientation]

        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"orientation": orientation})
        return obs, info


@register_env("GraspSingleCustomizableYCB-v0", max_episode_steps=80)
class GraspSingleCustomInSceneYCBEnv(GraspSingleCustomInSceneEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/ycb-dataset"
    DEFAULT_MODEL_JSON = "info_ycb.json"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@register_env("GraspSingleCustomizableNoOverlay-v0", max_episode_steps=80)
class GraspSingleCustomInSceneNoOverlayEnv(GraspSingleCustomInSceneEnv):
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret["control_mode"] = (
            "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        )
        ret["scene_name"] = "google_pick_coke_can_1_v4"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_cameras"] = ["overhead_camera"]

        return ret