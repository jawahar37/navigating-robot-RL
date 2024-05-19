import isaacgym

assert isaacgym
import torch
import numpy as np

import math
from isaacgym.torch_utils import *



import glob
import pickle as pkl

from go1_gym.envs import *

from go1_gym.envs.navigation.navigation_robot_config import Cfg
from go1_gym.envs.go1.position_tracking import PositionTrackingEasyEnv

from tqdm import tqdm
from go1_gym.envs.rewards.corl_rewards import quat_vector_cosine

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action
    
    return policy    


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[-1]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)
    
    Cfg.env.num_envs = 1

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = PositionTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    label = "navigation-policy/selected-run/main_navigation"
    num_eval_steps = 1000

    env, policy = load_env(label, headless=headless)
    
    obs = env.reset()
    print(obs)
    frames = []

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)

        obs, rew, done, info = env.step(actions)

        frames.append(env.render())

        
        position = env.base_pos[0].cpu().numpy()
        velocity = env.locomotion_env.base_lin_vel[0].cpu().numpy()
        if not i%20:
            print(f'position: {position[0]}, velocity: {velocity}')

    if len(frames) > 0:
        print("LOGGING VIDEO")
        logger.save_video(frames, "nav_play_demo.mp4", fps=1 / env.dt)


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)