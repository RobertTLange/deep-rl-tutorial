import argparse
import math
import random
import pandas as pd
import numpy as np
from collections import deque
import gym

import torch
import torch.autograd as autograd


def command_line_dqn():
    parser = argparse.ArgumentParser()
    # General logging/saving and device arguments
    parser.add_argument('-roll_upd', '--ROLLOUT_EVERY', action="store",
                        default=20, type=int,
                        help='Rollout test performance after # batch updates.')
    parser.add_argument('-n_roll', '--NUM_ROLLOUTS', action="store",
                        default=5, type=int,
                        help='# rollouts for tracking learning progrees')
    parser.add_argument('-n_runs', '--RUN_TIMES', action="store",
                        default=1, type=int,
                        help='# Times to run agent learning')
    parser.add_argument('-n_upd', '--NUM_UPDATES', action="store",
                        default=8000, type=int,
                        help='# SGD updates/iterations to train for')
    parser.add_argument('-max_steps', '--MAX_STEPS', action="store",
                        default=1000, type=int,
                        help='Max # of steps before episode terminated')
    parser.add_argument('-v', '--VERBOSE', action="store_true", default=False,
                        help='Get training progress printed out')
    parser.add_argument('-print', '--PRINT_EVERY', action="store",
                        default=500, type=int,
                        help='#Episodes after which to print if verbose.')
    parser.add_argument('-s', '--SAVE', action="store_true",
                        default=False, help='Save final agents and log')
    parser.add_argument('-device', '--device_id', action="store",
                        default=0, type=int, help='Device id on which to train')
    parser.add_argument('-fname', '--SAVE_FNAME', action="store",
                        default="temp", type=str, help='Filename to which to save logs')
    # Network architecture arguments
    parser.add_argument('-input', '--INPUT_DIM', action="store",
                        default=1200, type=int, help='Input Dimension')
    parser.add_argument('-hidden', '--HIDDEN_SIZE', action="store",
                        default=128, type=int, help='Hidden Dimension')
    parser.add_argument('-num_actions', '--NUM_ACTIONS', action="store",
                        default=4, type=int, help='Number of Actions')

    parser.add_argument('-gamma', '--GAMMA', action="store",
                        default=0.9, type=float,
                        help='Discount factor')
    parser.add_argument('-l_r', '--L_RATE', action="store", default=0.001,
                        type=float, help='Save network and learning stats after # epochs')
    parser.add_argument('-train_batch', '--TRAIN_BATCH_SIZE', action="store",
                        default=32, type=int, help='# images in training batch')

    parser.add_argument('-soft_tau', '--SOFT_TAU', action="store",
                        default=0., type=float,
                        help='Polyak Averaging tau for target network update')
    parser.add_argument('-update_upd', '--UPDATE_EVERY', action="store",
                        default=100, type=int,
                        help='Update target network after # batch updates')

    parser.add_argument('-e_start', '--EPS_START', action="store", default=1,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_stop', '--EPS_STOP', action="store", default=0.01,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_decay', '--EPS_DECAY', action="store", default=100,
                        type=float, help='Start Exploration Rate')

    parser.add_argument('-p', '--PER', action="store_true", default=False,
                        help='Perform prioritized experience replay sampling update.')
    parser.add_argument('-b_start', '--BETA_START', action="store", default=0.4,
                        type=float, help='Initial beta to start learning with.')
    parser.add_argument('-b_steps', '--BETA_STEPS', action="store", default=2000,
                        type=int, help='Number of steps until which beta is annealed to 1.')
    parser.add_argument('-alpha', '--ALPHA', action="store", default=0.6,
                        type=float, help='Temperature is priority Boltzmann distribution.')

    parser.add_argument('-agent', '--AGENT', action="store",
                        default="MLP-DQN", type=str, help='Agent model')
    parser.add_argument('-d', '--DOUBLE', action="store_true", default=False,
                        help='Perform double Q-Learning update.')
    parser.add_argument('-capacity', '--CAPACITY', action="store",
                        default=2000, type=int, help='Storage capacity of ER buffer')

    return parser.parse_args()


def compute_td_loss(agents, optimizer, replay_buffer, beta,
                    TRAIN_BATCH_SIZE, GAMMA, Variable, TRAIN_DOUBLE):

    if beta is None:
        obs, acts, reward, next_obs, done = replay_buffer.sample(TRAIN_BATCH_SIZE)
    else:
        obs, acts, reward, next_obs, done, indices, weights = replay_buffer.sample(TRAIN_BATCH_SIZE, beta)

    pyT = lambda array: Variable(torch.FloatTensor(array))
    obs = np.float32([ob.flatten() for ob in obs])
    next_obs = np.float32([next_ob.flatten() for next_ob in next_obs])
    obs, next_obs, reward = pyT(obs), pyT(next_obs), pyT(reward)
    action = Variable(torch.LongTensor(acts))
    done = Variable(torch.FloatTensor(done))

    if beta is not None:
        weights = pyT(weights)

    q_values = agents["current"](obs)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    if TRAIN_DOUBLE:
        next_q_values = agents["current"](next_obs)
        next_q_state_values = agents["target"](obs)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    else:
        next_q_values = agents["target"](next_obs)
        next_q_value = next_q_values.max(1)[0]

    expected_q_value = reward + GAMMA* next_q_value * (1 - done)
    loss = (q_value - expected_q_value.detach()).pow(2)

    if beta is not None:
        loss *= weights
        prios = loss + 1e-5
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    loss = loss.mean()

    # Perform optimization step for agent
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(agents["current"].parameters(), 0.5)
    optimizer.step()
    return loss
