import torch
import torch.nn as nn
import torch.autograd as autograd

import gym
import gridworld

import copy
import time
import math
import random
import numpy as np
import pandas as pd
from collections import deque

import torch.multiprocessing as mp

# Set device config variables
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


def init_weights(m):
    # Xavier initialization weights in network
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class ReplayBuffer(object):
    def __init__(self, capacity, record_macros=False):
        self.buffer = deque(maxlen=capacity)

    def push(self, ep_id, step, state, action,
             reward, next_state, done):
        self.buffer.append((ep_id, step, state, action, reward, next_state, done))

    def sample(self, batch_size):
        ep_id, step, state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done

    def num_episodes(self):
        return len(np.unique(np.array(self.buffer)[:,0]))

    def __len__(self):
        return len(self.buffer)


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, ep_id, step, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((ep_id, step, state, action,
                                reward, next_state, done))
        else:
            self.buffer[self.pos] = (ep_id, step, state, action,
                                     reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity: prios = self.priorities
        else: prios = self.priorities[:self.pos]

        probs  = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        N = len(self.buffer)
        weights  = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        batch = list(zip(*samples))
        states, next_states = np.stack(batch[2]), np.stack(batch[5])
        actions, rewards, dones = batch[3], batch[4], batch[6]
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


def epsilon_by_update(opt_counter, epsilon_start, epsilon_final, epsilon_decay):
    # Exponentially decaying exploration strategy
    eps = min(epsilon_final, epsilon_start + opt_counter * (epsilon_final - epsilon_start) / epsilon_decay)
    return eps


def beta_by_update(opt_counter, beta_start=0.4, beta_steps=2000):
    # Linearly increasing temperature parameter
    beta = min(1.0, beta_start + opt_counter * (1.0 - beta_start) / beta_steps)
    return beta


def update_target(current_model, target_model):
    # Transfer parameters from current model to target model
    target_model.load_state_dict(current_model.state_dict())


def polyak_update_target(current_model, target_model, soft_tau):
    for target_param, current_param in zip(target_model.parameters(),
                                           current_model.parameters()):
        target_param.data.copy_(
            target_param.data * (1. - soft_tau) + current_param.data * soft_tau
        )


def get_logging_stats(opt_counter, agent, GAMMA,
                      NUM_ROLLOUTS, MAX_STEPS, AGENT):
    steps = []
    rew = []

    for i in range(NUM_ROLLOUTS):
        step_temp, reward_temp, buffer = rollout_episode(agent, GAMMA,
                                                         MAX_STEPS, AGENT)
        steps.append(step_temp)
        rew.append(reward_temp)

    steps = np.array(steps)
    rew = np.array(rew)

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median",
                                         "rew_10th_p", "rew_90th_p"])

    steps_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                        "steps_median",
                                        "steps_10th_p", "steps_90th_p"])

    reward_stats.loc[0] = [opt_counter, rew.mean(), rew.std(), np.median(rew),
                           np.percentile(rew, 10), np.percentile(rew, 90)]

    steps_stats.loc[0] = [opt_counter, steps.mean(), steps.std(), np.median(steps),
                         np.percentile(steps, 10), np.percentile(steps, 90)]

    return reward_stats, steps_stats


def rollout_episode(agent, GAMMA, MAX_STEPS, AGENT):
    env = gym.make("dense-v0")
    # Rollout the policy for a single episode - greedy!
    replay_buffer = ReplayBuffer(capacity=5000)

    obs = env.reset()
    episode_rew = 0
    steps = 0
    GAMMA = 0.95

    while steps < MAX_STEPS:
        action = agent["current"].act(obs.flatten(), epsilon=0.05)
        next_obs, reward, done, _ = env.step(action)
        steps += 1

        replay_buffer.push(0, steps, obs, action,
                           reward, next_obs, done)

        obs = next_obs

        episode_rew += GAMMA**(steps - 1) * reward
        if done:
            break
    return steps, episode_rew, replay_buffer.buffer


def run_multiple_times(args, run_fct):
    cpu_count = mp.cpu_count()
    gpu_count = torch.cuda.device_count()

    # Clone arguments into list & Distribute workload across GPUs
    args_across_workers = [copy.deepcopy(args) for r in range(args.RUN_TIMES)]
    if gpu_count > 0:
        gpu_counter = 0
        for r in range(args.RUN_TIMES):
            args_across_workers[r].device_id = gpu_counter
            gpu_counter += 1
            if gpu_counter > gpu_count-1:
                gpu_counter = 0

    # Execute different runs/random seeds in parallel
    pool = mp.Pool(cpu_count-1)
    df_across_runs = pool.map(run_fct, args_across_workers)
    pool.close()

    # Post process results
    df_concat = pd.concat(df_across_runs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means, df_stds = by_row_index.mean(), by_row_index.std()
    if args.SAVE:
        df_means.to_csv("logs/" + args.SAVE_FNAME + ".csv")
    return df_means, df_stds
