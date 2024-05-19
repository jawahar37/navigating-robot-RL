import isaacgym
assert isaacgym
import torch


from go1_gym.envs.go1.position_tracking import PositionTrackingEasyEnv

from ml_logger import logger

from go1_gym_learn.ppo_cse_nav import Runner
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym_learn.ppo_cse_nav.actor_critic import AC_Args
from go1_gym_learn.ppo_cse_nav.ppo import PPO_Args
from go1_gym_learn.ppo_cse_nav import RunnerArgs

final_log_path = ''


def train_go1(headless=True):
  from go1_gym.envs.navigation.navigation_robot_config import Cfg

  env = PositionTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=Cfg)

  # log the experiment parameters
  logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                    Cfg=vars(Cfg))

  env = HistoryWrapper(env)
  gpu_id = 0
  runner = Runner(env, final_log_path, device=f"cuda:{gpu_id}")
  num_learning_iterations = 12000
  runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from go1_gym import MINI_GYM_ROOT_DIR
    from datetime import datetime

    stem = Path(__file__).stem
    current_datetime = datetime.utcnow()
    log_file_path = f"navigation-policy/{current_datetime.strftime('%Y-%m-%d')}/{stem}/{current_datetime.strftime('%H%M%S.%f')}"
    logger.configure(logger.utcnow(log_file_path),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    final_log_path = f"{MINI_GYM_ROOT_DIR}/runs/{log_file_path}"
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_1/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_2/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_orientation_control/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_pos/mean
                  xKey: iterations
                - yKey: train/episode/command_area_trot/mean
                  xKey: iterations
                - yKey: train/episode/max_terrain_height/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                - yKey: adaptation_loss/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    # to see the environment rendering, set headless=False
    train_go1(headless=True)
