import gym
import time
import numpy as np
import pandas as pd
import gridworld

import torch
import torch.autograd as autograd
import torch.multiprocessing as mp

from dqn import MLP_DQN, MLP_DuelingDQN, init_dqn
from dqn_helpers import command_line_dqn, compute_td_loss
from general_helpers import epsilon_by_update, beta_by_update
from general_helpers import update_target, polyak_update_target
from general_helpers import ReplayBuffer, NaivePrioritizedBuffer
from general_helpers import get_logging_stats, run_multiple_times

torch.manual_seed(0)
np.random.seed(0)

def run_dqn_learning(args):
    log_template = "Step {:>2} | T {:.1f} | Median R {:.1f} | Mean R {:.1f} | Median S {:.1f} | Mean S {:.1f}"

    # Set the GPU device on which to run the agent
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        torch.cuda.set_device(args.device_id)
    else:
        print("USING CPU")
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
    start = time.time()

    if args.DOUBLE: TRAIN_DOUBLE = True
    else: TRAIN_DOUBLE = False

    # Setup agent, replay replay_buffer, logging stats df
    if args.AGENT == "MLP-DQN":
        agents, optimizer = init_dqn(MLP_DQN, args.L_RATE, USE_CUDA,
                                     args.INPUT_DIM, args.HIDDEN_SIZE,
                                     args.NUM_ACTIONS)
    elif args.AGENT == "MLP-DUELING":
        agents, optimizer = init_dqn(MLP_DuelingDQN, args.L_RATE, USE_CUDA,
                                     args.INPUT_DIM, args.HIDDEN_SIZE,
                                     args.NUM_ACTIONS)

    if not args.PER:
        replay_buffer = ReplayBuffer(capacity=args.CAPACITY)
    else:
        replay_buffer = NaivePrioritizedBuffer(capacity=args.CAPACITY,
                                               prob_alpha=args.ALPHA)

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median", "rew_10th_p", "rew_90th_p"])

    step_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                       "steps_median", "steps_10th_p", "steps_90th_p"])

    # Initialize optimization update counter and environment
    opt_counter = 0
    env = gym.make("dense-v0")
    ep_id = 0
    # RUN TRAINING LOOP OVER EPISODES
    while opt_counter < args.NUM_UPDATES:
        obs = env.reset()
        steps = 0

        while steps < args.MAX_STEPS:
            epsilon = epsilon_by_update(opt_counter + 1, args.EPS_START,
                                         args.EPS_STOP, args.EPS_DECAY)
            if args.PER:
                beta = beta_by_update(opt_counter + 1, args.BETA_START,
                                      args.BETA_STEPS)
            else:
                beta = None

            action = agents["current"].act(obs.flatten(), epsilon)
            next_obs, rew, done, _  = env.step(action)
            steps += 1

            # Push transition to ER Buffer
            replay_buffer.push(ep_id, steps, obs, action,
                               rew, next_obs, done)

            if len(replay_buffer) > args.TRAIN_BATCH_SIZE:
                opt_counter += 1
                loss = compute_td_loss(agents, optimizer, replay_buffer, beta,
                                       args.TRAIN_BATCH_SIZE, args.GAMMA, Variable,
                                       TRAIN_DOUBLE)

                if args.SOFT_TAU != 0:
                    polyak_update_target(agents["current"], agents["target"],
                                         args.SOFT_TAU)


            # Go to next episode if current one terminated or update obs
            if done: break
            else: obs = next_obs

            # On-Policy Rollout for Performance evaluation
            if (opt_counter+1) % args.ROLLOUT_EVERY == 0:
                r_stats, s_stats = get_logging_stats(opt_counter, agents,
                                                     args.GAMMA, args.NUM_ROLLOUTS,
                                                     args.MAX_STEPS, args.AGENT)
                reward_stats = pd.concat([reward_stats, r_stats], axis=0)
                step_stats = pd.concat([step_stats, s_stats], axis=0)

            if args.SOFT_TAU == 0 and (opt_counter+1) % args.UPDATE_EVERY == 0:
                update_target(agents["current"], agents["target"])

            if args.VERBOSE and (opt_counter+1) % args.PRINT_EVERY == 0:
                stop = time.time()
                print(log_template.format(opt_counter+1, stop-start,
                                          r_stats.loc[0, "rew_median"],
                                          r_stats.loc[0, "rew_mean"],
                                          s_stats.loc[0, "steps_median"],
                                          s_stats.loc[0, "steps_mean"]))
                start = time.time()

        ep_id += 1

    # Save the logging dataframe
    df_to_save = pd.concat([reward_stats, step_stats], axis=1)
    df_to_save = df_to_save.loc[:,~df_to_save.columns.duplicated()]
    df_to_save = df_to_save.reset_index()

    # Finally save all results!
    # if args.SAVE:
    #     torch.save(agents["current"].state_dict(), "agents/" + str(args.NUM_UPDATES) + "_" + args.AGENT)
    #     df_to_save.to_csv("results/"  + args.AGENT + "_" + args.STATS_FNAME)
    return df_to_save


if __name__ == "__main__":
    mp.set_start_method('forkserver')
    args = command_line_dqn()

    if args.RUN_TIMES == 1:
        print("START RUNNING {} AGENT LEARNING FOR 1 TIME".format(args.AGENT))
        run_dqn_learning(args)
    else:
        start_t = time.time()
        run_multiple_times(args, run_dqn_learning)
        print("Done - {} experiment after {:.2f} secs".format(args.SAVE_FNAME,
                                                              time.time() - start_t))
