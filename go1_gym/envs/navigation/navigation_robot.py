# Navigation Robot.py 

# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from typing import Dict
import pickle as pkl

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.base_task import BaseTask
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from go1_gym.utils.terrain import Terrain
# from .legged_robot_config import Cfg
from go1_gym.envs.navigation.navigation_robot_config import Cfg

import glob
import os
from pathlib import Path
from go1_gym import MINI_GYM_ROOT_DIR

from tqdm import tqdm

class NavigationRobot:
    def __init__(self, cfg: Cfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None,
                 initial_dynamics_dict=None):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.sim_params = sim_params
        self.init_done = False
        self.initial_dynamics_dict = initial_dynamics_dict
        if eval_cfg is not None: self._parse_cfg(eval_cfg)
        self._parse_cfg(self.cfg)

        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(sim_device)
        
        self.headless = headless
        
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline: self.device = sim_device
        else: self.device = 'cpu' 
        
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = self.sim_device_id

        self.num_obs = cfg.env.num_observations    
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions  
        
        if eval_cfg is not None:
            self.num_eval_envs = eval_cfg.env.num_envs
            self.num_train_envs = cfg.env.num_envs
            self.num_envs = self.num_eval_envs + self.num_train_envs
        else:
            self.num_eval_envs = 0
            self.num_train_envs = cfg.env.num_envs
            self.num_envs = cfg.env.num_envs 
        

        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_neg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                              dtype=torch.float)
        
        self.extras = {}
        self.create_sim() # loads the locomotion env and policy
        
        self.enable_viewer_sync = True
        self.viewer = None
        self.reward_plot_tracker = {}
        self.succes_rate_counter = []

        self._init_buffers()

        self._prepare_reward_function()
        self.init_done = True
        self.record_now = False
        self.record_eval_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0

        if  cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:# implemented only for plane now
                 self.custom_origins = False

        self.env_origins = self.locomotion_env.env_origins

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()

        gaits = {"pronking": [0, 0, 0],
            "trotting": [0.5, 0, 0],
            "bounding": [0, 0.5, 0],
            "pacing": [0, 0, 0.5]}
        body_height_cmd = 0.0
        step_frequency_cmd = 3.0
        gait = torch.tensor(gaits["trotting"])
        footswing_height_cmd = 0.08
        pitch_cmd = 0.0
        roll_cmd = 0.0
        stance_width_cmd = 0.25
        import copy
        # print("step :self.locomotion_env.obs ", self.locomotion_env.obs)
        with torch.no_grad():
            locomotion_actions = self.locomotion_policy(self.locomotion_env.obs)
            # print("sttep: locomotion_actions :", locomotion_actions)
        self.locomotion_env.commands[:, 0] = self.actions[:,0]
        self.locomotion_env.commands[:, 1] = self.actions[:,1]
        self.locomotion_env.commands[:, 2] = self.actions[:,2] 
        self.locomotion_env.commands[:, 3] = body_height_cmd
        self.locomotion_env.commands[:, 4] = step_frequency_cmd
        self.locomotion_env.commands[:, 5:8] = gait
        self.locomotion_env.commands[:, 8] = 0.5
        self.locomotion_env.commands[:, 9] = footswing_height_cmd
        self.locomotion_env.commands[:, 10] = pitch_cmd
        self.locomotion_env.commands[:, 11] = roll_cmd
        self.locomotion_env.commands[:, 12] = stance_width_cmd
        self.locomotion_env.obs, self.locomotion_env.rew, self.locomotion_env.done, self.locomotion_env.info = self.locomotion_env.step(locomotion_actions)
        
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.root_states = self.locomotion_env.root_states.clone()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]


        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
                
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.extras["train/success_counts"] = self.destination_buf
        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        

    def check_termination(self):
        """ Check if environments need to be reset
        """
        
        self.reset_buf = self.locomotion_env.reset_buf.clone()        
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length  # no terminal reward for time-outs
        self.destination_buf = self.locomotion_env.base_pos[:,0] - self.locomotion_env.env_origins[:,0] >= 3
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.destination_buf


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        self._call_train_eval(self._reset_root_states, env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        # self.last_dof_vel[env_ids] = 0.
        # self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["train/episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][train_env_ids])
                self.episode_sums[key][train_env_ids] = 0.
        eval_env_ids = env_ids[env_ids >= self.num_train_envs]
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}
            for key in self.episode_sums.keys():
                # save the evaluation rollout result if not already saved
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf[:self.num_train_envs]

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        self.episode_sums["total"] += self.rew_buf
        

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        self.obs_buf = torch.cat((
                                    (self.base_pos[:] - self.env_origins[:]) * self.obs_scales.lin_pos,
                                    self.actions
                                    ), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.last_actions), dim=-1)

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            self.obs_buf = torch.cat((self.obs_buf,
                                      heading), dim=-1)
        if self.cfg.env.observe_obstacle_states:
            self.obs_buf = torch.cat((self.obs_buf , 
            torch.tensor(self.locomotion_env.obstacle_s_wall_pos_x, device = self.device).unsqueeze(1) , 
            torch.tensor(self.locomotion_env.obstacle_s_wall_pos_y , device = self.device).unsqueeze(1), 
            torch.tensor(self.locomotion_env.obstacle_s_length , device = self.device).unsqueeze(1) ,
            torch.tensor(self.locomotion_env.obstacle_s_thickness , device = self.device).unsqueeze(1) ) , dim = -1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device) 
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device) 
        assert self.privileged_obs_buf.shape[
                   1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    def create_sim(self):
        self._create_envs()

    # ------------- Callbacks --------------
    def _call_train_eval(self, func, env_ids):

        env_ids_train = env_ids[env_ids < self.num_train_envs]
        env_ids_eval = env_ids[env_ids >= self.num_train_envs]

        ret, ret_eval = None, None

        if len(env_ids_train) > 0:
            ret = func(env_ids_train, self.cfg)
        if len(env_ids_eval) > 0:
            ret_eval = func(env_ids_eval, self.eval_cfg)
            if ret is not None and ret_eval is not None: ret = torch.cat((ret, ret_eval), axis=-1)

        return ret

    def _reset_root_states(self, env_ids, cfg):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.locomotion_env.reset_root_states(env_ids , self.locomotion_Cfg)
        
        if not self.custom_origins: #only for a plane atm
            self.root_states[env_ids] = self.locomotion_env.root_states[env_ids].clone()

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec = torch.cat((
                                torch.ones(3) * noise_scales.lin_pos * noise_level * self.obs_scales.lin_pos, # pos noise
                                torch.zeros(self.num_actions),
                                ), dim=0)

        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(self.num_actions)
                                   ), dim=0)
        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1),
                                   ), dim=0)
        if self.cfg.env.observe_obstacle_states:
            noise_vec = torch.cat((noise_vec, torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1)) , dim = 0)


        noise_vec = noise_vec.to(self.device)

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        self.root_states = self.locomotion_env.root_states.clone()
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.base_quat = self.root_states[:self.num_envs, 3:7]

        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()

        self.common_step_counter = 0
        self.extras = {}

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # , self.eval_cfg)
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
 
        from go1_gym.envs.rewards.corl_rewards import CoRLRewards
        reward_containers = {"CoRLRewards": CoRLRewards}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)

    def load_locomotion_policy_f(self,logdir):
        body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
        import os
        adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

        def locomotion_policy_f(obs, info={}):
            i = 0
            latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
            action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
            info['latent'] = latent
            return action

        return locomotion_policy_f

    def load_locomotion_env(self, label, headless=False):
        from go1_gym.envs.base.legged_robot_config import Cfg as locomotion_Cfg
        from go1_gym.envs.go1.go1_config import config_go1
        from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv       
        self.locomotion_Cfg = locomotion_Cfg
        dirs = glob.glob(f"../runs/{label}/*")
        logdir = sorted(dirs)[-1]

        with open(logdir + "/parameters.pkl", 'rb') as file:
            pkl_cfg = pkl.load(file)
            print(pkl_cfg.keys())
            cfg = pkl_cfg["Cfg"]
            print(cfg.keys())

            for key, value in cfg.items():
                if hasattr(self.locomotion_Cfg, key):
                    for key2, value2 in cfg[key].items():
                        setattr(getattr(self.locomotion_Cfg, key), key2, value2)

        # turn off DR for evaluation script
        self.locomotion_Cfg.domain_rand.push_robots = False
        self.locomotion_Cfg.domain_rand.randomize_friction = False
        self.locomotion_Cfg.domain_rand.randomize_gravity = False
        self.locomotion_Cfg.domain_rand.randomize_restitution = False
        self.locomotion_Cfg.domain_rand.randomize_motor_offset = False
        self.locomotion_Cfg.domain_rand.randomize_motor_strength = False
        self.locomotion_Cfg.domain_rand.randomize_friction_indep = False
        self.locomotion_Cfg.domain_rand.randomize_ground_friction = False
        self.locomotion_Cfg.domain_rand.randomize_base_mass = False
        self.locomotion_Cfg.domain_rand.randomize_Kd_factor = False
        self.locomotion_Cfg.domain_rand.randomize_Kp_factor = False
        self.locomotion_Cfg.domain_rand.randomize_joint_friction = False
        self.locomotion_Cfg.domain_rand.randomize_com_displacement = False
        # locomotion_Cfg.domain_rand.randomize_rigids_after_start = False

        self.locomotion_Cfg.env.num_recording_envs = 1
        self.locomotion_Cfg.env.num_train_envs = self.num_train_envs
        self.locomotion_Cfg.env.num_eval_envs = self.num_eval_envs
        self.locomotion_Cfg.env.num_envs = self.num_envs
        # locomotion_Cfg.record_video = True
        # locomotion_Cfg.record_eval_now = False
        self.locomotion_Cfg.recording_width_px = 368*2
        self.locomotion_Cfg.recording_height_px = 240*2
        self.locomotion_Cfg.recording_mode = "COLOR"

        self.locomotion_Cfg.terrain.num_rows = 5
        self.locomotion_Cfg.terrain.num_cols = 5
        self.locomotion_Cfg.terrain.border_size = 0
        self.locomotion_Cfg.terrain.center_robots = True
        self.locomotion_Cfg.terrain.center_span = 1
        self.locomotion_Cfg.terrain.teleport_robots = False

        self.locomotion_Cfg.domain_rand.lag_timesteps = 6
        self.locomotion_Cfg.domain_rand.randomize_lag_timesteps = True
        self.locomotion_Cfg.control.control_type = "actuator_net"

        self.locomotion_Cfg.terrain.x_init_range = 0
        self.locomotion_Cfg.terrain.y_init_range = 0
        self.locomotion_Cfg.terrain.yaw_init_range = 0
        self.locomotion_Cfg.terrain.mesh_type = 'plane'

        self.locomotion_Cfg.viewer.pos = [-0.4 , 0 , 3.5 ]
        self.locomotion_Cfg.viewer.lookat =  [1,0,0]

        from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper as locomotion_history_wrapper

        locomotion_env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=locomotion_Cfg)
        locomotion_env = locomotion_history_wrapper(locomotion_env)

        # load policy
        from ml_logger import logger
        from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

        locomotion_policy = self.load_locomotion_policy_f(logdir)

        self.locomotion_env = locomotion_env
        self.locomotion_policy = locomotion_policy

    def _create_envs(self):
        from ml_logger import logger
        from pathlib import Path
        from go1_gym import MINI_GYM_ROOT_DIR
        import glob
        import os

        label = "gait-conditioned-agility/2023-10-17/train"
        self.load_locomotion_env(label , headless=False )
        self.locomotion_env.obs = self.locomotion_env.reset()

        self.video_writer = self.locomotion_env.video_writer
        self.video_frames = self.locomotion_env.video_frames
        self.video_frames_eval = self.locomotion_env.video_frames_eval
        self.complete_video_frames = self.locomotion_env.complete_video_frames
        self.complete_video_frames_eval = self.locomotion_env.complete_video_frames_eval        


    def render(self, mode="rgb_array"):
        return self.locomotion_env.render()

    def start_recording(self):
        self.locomotion_env.start_recording()

    def start_recording_eval(self):
        self.locomotion_env.start_recording_eval()

    def pause_recording(self):
        self.locomotion_env.pause_recording()

    def pause_recording_eval(self):
        self.locomotion_env.pause_recording_eval()

    def get_complete_frames(self):
        return self.locomotion_env.get_complete_frames()
        
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.obs_scales
        self.reward_scales = vars(self.cfg.reward_scales)
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        cfg.command_ranges = vars(cfg.commands)
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            cfg.terrain.curriculum = False
        max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length
