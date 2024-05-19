import time
from collections import deque
import copy
import os

import torch
from ml_logger import logger
from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

from matplotlib import pyplot as plt
import numpy as np

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from go1_gym_learn.ppo.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = False


class Runner:

    def __init__(self, env, log_path, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env
        self.log_path = log_path

        plot_dir = f'{self.log_path}/plots'
        print(f'plot_dir: {plot_dir}')

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      ).to(self.device)

        if RunnerArgs.resume:
            # load pretrained weights from resume_path
            from ml_logger import ML_Logger
            loader = ML_Logger(root="http://escher.csail.mit.edu:8080",
                               prefix=RunnerArgs.resume_path)
            weights = loader.load_torch("checkpoints/ac_weights_last.pt")
            actor_critic.load_state_dict(state_dict=weights)

            if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
                # load curriculum state
                distributions = loader.load_pkl("curriculum/distribution.pkl")
                distribution_last = distributions[-1]["distribution"]
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0
        self.reward_plot_tracker_per_iteration = {}
        self.success_plot_tracker_per_iteration = []
        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        from ml_logger import logger
        # initialize writer
        assert logger.prefix, "you will overwrite the entire instrument server"

        logger.start('start', 'epoch', 'episode', 'run', 'step')

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rewards and success Plot tracking:
            Rewards_plot_tracker_per_runner_arg_steps = {}
            Succes_plot_tracker_per_runner_arg_steps = []
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    if eval_expert:
                        actions_eval = self.alg.actor_critic.act_teacher(obs_history[num_train_envs:],
                                                                         privileged_obs[num_train_envs:])
                    else:
                        actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:])
                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)
                    
                    for key in infos["train/episode"].keys():
                        if key in Rewards_plot_tracker_per_runner_arg_steps:
                            Rewards_plot_tracker_per_runner_arg_steps[key].extend([infos["train/episode"][key].item()])
                        else:
                            Rewards_plot_tracker_per_runner_arg_steps[key]= [infos["train/episode"][key].item()]
                    
                    Succes_plot_tracker_per_runner_arg_steps.extend([sum(infos["train/success_counts"].cpu().numpy().astype(int))])
                    # print(infos["train/success_counts"].cpu().numpy().astype(int))
                    # print(infos["body_pos"])

                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])

                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])

                    # if 'curriculum' in infos:

                    #     cur_reward_sum += rewards
                    #     cur_episode_length += 1

                    #     new_ids = (dones > 0).nonzero(as_tuple=False)

                    #     new_ids_train = new_ids[new_ids < num_train_envs]
                    #     rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                    #     lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                    #     cur_reward_sum[new_ids_train] = 0
                    #     cur_episode_length[new_ids_train] = 0

                    #     new_ids_eval = new_ids[new_ids >= num_train_envs]
                    #     rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                    #     lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                    #     cur_reward_sum[new_ids_eval] = 0
                    #     cur_episode_length[new_ids_eval] = 0

                    # if 'curriculum/distribution' in infos:
                    #     distribution = infos['curriculum/distribution']
           
                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

                # if it % curriculum_dump_freq == 0:
                #     logger.save_pkl({"iteration": it,
                #                      **caches.slot_cache.get_summary(),
                #                      **caches.dist_cache.get_summary()},
                #                     path=f"curriculum/info.pkl", append=True)

                #     if 'curriculum/distribution' in infos:
                #         logger.save_pkl({"iteration": it,
                #                          "distribution": distribution},
                #                          path=f"curriculum/distribution.pkl", append=True)

            for key in Rewards_plot_tracker_per_runner_arg_steps.keys():
                if key in self.reward_plot_tracker_per_iteration:
                    self.reward_plot_tracker_per_iteration[key].extend([sum(Rewards_plot_tracker_per_runner_arg_steps[key])/self.num_steps_per_env])
                else:
                    self.reward_plot_tracker_per_iteration[key] = [sum(Rewards_plot_tracker_per_runner_arg_steps[key])/self.num_steps_per_env]

            self.success_plot_tracker_per_iteration.extend([sum(Succes_plot_tracker_per_runner_arg_steps)])

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            logger.store_metrics(
                # total_time=learn_time - collection_time,
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                mean_decoder_loss=mean_decoder_loss,
                mean_decoder_loss_student=mean_decoder_loss_student,
                mean_decoder_test_loss=mean_decoder_test_loss,
                mean_decoder_test_loss_student=mean_decoder_test_loss_student,
                mean_adaptation_module_test_loss=mean_adaptation_module_test_loss
            )

            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
                self.log_plots(it)
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                logger.job_running()


            if it % RunnerArgs.save_interval == 0:
                with logger.Sync():
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = './tmp/legged_data'

                    os.makedirs(path, exist_ok=True)

                    adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                    adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                    traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    traced_script_adaptation_module.save(adaptation_module_path)

                    body_path = f'{path}/body_latest.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(body_path)

                    logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
                    logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

            self.current_learning_iteration += num_learning_iterations

            # print("*******************************************************************************\n")

        with logger.Sync():
            logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            path = './tmp/legged_data'

            os.makedirs(path, exist_ok=True)

            adaptation_module_path = f'{path}/adaptation_module_latest.jit'
            adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            traced_script_adaptation_module.save(adaptation_module_path)

            body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

            logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)


    def log_plots(self, it):
        training_rewards = self.reward_plot_tracker_per_iteration
        success_rate = self.success_plot_tracker_per_iteration
        num_rewards = len(training_rewards.keys())
        num_learning_iterations = len(success_rate)
        
        plot_dir = f'{self.log_path}/plots'
        print(f'plot_dir: {plot_dir}')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        # plot rewards
        plt.rcParams.update({'font.size': 7})
        fig, axs = plt.subplots(num_rewards, 1, figsize=(18, 15))
        line_space_intervals = num_learning_iterations# if num_learning_iterations<= 1000 else 1000 

        for i,x in enumerate(training_rewards.keys()):
            axs[i].plot(np.linspace(0, num_learning_iterations, line_space_intervals), training_rewards[x], color='black', linestyle="-", label="Reward")
            # axs[0].legend()
            axs[i].set_title(x)
            axs[i].set_ylabel("Direct Reward values")
            axs[i].grid(True)

        
        fig.savefig(f"{plot_dir}/{it}_training_rewards.jpg")

        # plot rewards (cumulative)
        fig, axs = plt.subplots(num_rewards, 1, figsize=(18, 15))
        line_space_intervals = num_learning_iterations# if num_learning_iterations<= 1000 else 1000 

        for i,x in enumerate(training_rewards.keys()):
            axs[i].plot(np.linspace(0, num_learning_iterations, line_space_intervals), np.cumsum(training_rewards[x]), color='black', linestyle="-", label="Reward")
            # axs[0].legend()
            axs[i].set_title(x)
            axs[i].set_ylabel("Direct Reward values")
            axs[i].grid(True)

        
        fig.savefig(f"{plot_dir}/{it}_training_rewards_sums.jpg")

        #plot success rates
        print("success_rate_per_iteration", success_rate)
        plt.figure(figsize=(18, 6))
        plt.plot(np.linspace(0, num_learning_iterations, line_space_intervals), success_rate)
        plt.xlabel('Iterations')
        plt.ylabel('Success Rates')
        plt.title('Success per Iterations')

        # Save the plot as an image file (e.g., PNG, JPEG, PDF)
        plt.savefig(f"{plot_dir}/{it}_success_per_iteration.jpg")

        plt.figure(figsize=(18, 6))
        plt.plot(np.linspace(0, num_learning_iterations, line_space_intervals), np.cumsum(success_rate))
        plt.xlabel('Iterations')
        plt.ylabel('Success Rates')
        plt.title('Success (cumulative)')

        # Save the plot as an image file (e.g., PNG, JPEG, PDF)
        plt.savefig(f"{plot_dir}/{it}_success_sums.jpg")

    def log_video(self, it):
        # print("-------THIS IS INSIDE LOG VIDEO------")
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
