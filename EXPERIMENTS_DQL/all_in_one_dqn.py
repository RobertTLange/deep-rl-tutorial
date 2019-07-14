import math
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

import gym
import gridworld
from dqn_helpers import command_line_dqn

# Set the GPU device on which to run the agent
USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
else: print("USING CPU")
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class MLP_DQN(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS):
        super(MLP_DQN, self).__init__()
        self.action_space_size = NUM_ACTIONS
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, self.action_space_size)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.action_space_size)
        return action

class ReplayBuffer(object):
    def __init__(self, capacity, record_macros=False):
        self.buffer = deque(maxlen=capacity)

    def push(self, ep_id, step, state, action,
             reward, next_state, done):
        self.buffer.append((ep_id, step, state, action, reward, next_state, done))

    def sample(self, batch_size):
        _, step, s, act, rew, next_s, done = zip(*random.sample(self.buffer,
                                                                batch_size))
        return np.stack(s), act, rew, np.stack(next_s), done

    def __len__(self):
        return len(self.buffer)

def init_dqn(model, L_RATE, USE_CUDA, INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS):
    agents = {"current": model(INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS),
              "target": model(INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS)}

    if USE_CUDA:
        agents["current"] = agents["current"].cuda()
        agents["target"] = agents["target"].cuda()

    optimizers = optim.Adam(params=agents["current"].parameters(), lr=L_RATE)
    return agents, optimizers

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def epsilon_by_episode(eps_id, epsilon_start, epsilon_final, epsilon_decay):
    eps = (epsilon_final + (epsilon_start - epsilon_final)
           * math.exp(-1. * eps_id / epsilon_decay))
    return eps

def run_dqn_learning(args):
    agents, optimizer = init_dqn(MLP_DQN, args.L_RATE, USE_CUDA,
                                 args.INPUT_DIM, args.HIDDEN_SIZE, args.NUM_ACTIONS)
    replay_buffer = ReplayBuffer(capacity=args.CAPACITY)

    opt_counter = 0
    env = gym.make("dense-v0")
    ep_id = 0

    while opt_counter < args.NUM_UPDATES:
        epsilon = epsilon_by_episode(ep_id + 1, args.EPS_START, args.EPS_STOP,
                                     args.EPS_DECAY)
        obs = env.reset()
        steps = 0
        while steps < args.MAX_STEPS:
            action = agents["current"].act(obs.flatten(), epsilon)
            next_obs, rew, done, _  = env.step(action)
            steps += 1

            # Push transition to ER Buffer
            replay_buffer.push(ep_id, steps, obs, action,
                               rew, next_obs, done)

            if len(replay_buffer) > args.TRAIN_BATCH_SIZE:
                opt_counter += 1
                loss = compute_td_loss(agents, optimizer, replay_buffer,
                                       args.TRAIN_BATCH_SIZE, args.GAMMA, Variable)

            # Go to next episode if current one terminated or update obs
            if done: break
            else: obs = next_obs

            if (opt_counter+1) % args.UPDATE_EVERY == 0:
                update_target(agents["current"], agents["target"])
        ep_id += 1
    return

def compute_td_loss(agents, optimizer, replay_buffer,
                    TRAIN_BATCH_SIZE, GAMMA, Variable):
    obs, acts, reward, next_obs, done = replay_buffer.sample(TRAIN_BATCH_SIZE)

    pyT = lambda array: Variable(torch.FloatTensor(array))
    obs = np.float32([ob.flatten() for ob in obs])
    next_obs = np.float32([next_ob.flatten() for next_ob in next_obs])
    obs, next_obs, reward = pyT(obs), pyT(next_obs), pyT(reward)
    action = Variable(torch.LongTensor(acts))
    done = Variable(torch.FloatTensor(done))

    q_value = agents["current"](obs).gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = agents["target"](next_obs).max(1)[0]

    expected_q_value = reward + GAMMA* next_q_value * (1 - done)
    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(agents["current"].parameters(), 0.5)
    optimizer.step()
    return loss


class MLP_DuelingDQN(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS):
        super(MLP_DuelingDQN, self).__init__()
        self.action_space_size = NUM_ACTIONS
        self.feature = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_SIZE),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, self.action_space_size)
        )
        self.value = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.action_space_size)
        return action

if __name__ == "__main__":
    args = command_line_dqn()
    run_dqn_learning(args)
