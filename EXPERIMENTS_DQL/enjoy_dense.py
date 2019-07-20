import gym
import matplotlib.pyplot as plt
import numpy as np
import gridworld
from dqn import init_dqn, MLP_DQN
from pycolab import rendering
import torch


def rgb_rescale(v):
    return v/255


COLOUR_FG = {' ': tuple([rgb_rescale(v) for v in (123, 132, 150)]), # Background
             '$': tuple([rgb_rescale(v) for v in (214, 182, 79)]),  # Coins
             '@': tuple([rgb_rescale(v) for v in (66, 6, 13)]),     # Poison
             '#': tuple([rgb_rescale(v) for v in (119, 107, 122)]), # Walls of the maze
             'P': tuple([rgb_rescale(v) for v in (153, 85, 74)]),   # Player
             'a': tuple([rgb_rescale(v) for v in (107, 132, 102)]), # Patroller A
             'b': tuple([rgb_rescale(v) for v in (107, 132, 102)])} # Patroller B


def converter(obs):
    converter = rendering.ObservationToArray(COLOUR_FG, permute=(0,1,2))
    converted = np.swapaxes(converter(obs), 1, 2).T
    return converted


def main(LOAD_CKPT):
    env = gym.make("dense-v0")

    USE_CUDA = torch.cuda.is_available()
    agents, optimizer = init_dqn(MLP_DQN, 0, USE_CUDA,
                                   1200, 128, 4, LOAD_CKPT)

    obs, screen_obs = env.reset_with_render()
    done = False
    episode_rew = 0
    converted = converter(screen_obs)
    my_plot = plt.imshow(converted)
    steps = 0
    while not done:
        #obs, rew, done, _ , screen_obs = env.step_with_render(act(obs)[0])
        #obs, rew, done, _ , screen_obs = env.step_with_render(env.action_space.sample())
        action = agents["current"].act(obs.flatten(), epsilon=0.05)
        obs, rew, done, _ , screen_obs = env.step_with_render(action)
        converted = converter(screen_obs)
        plt.ion()
        my_plot.autoscale()
        my_plot.set_data(converted)
        plt.pause(.05)
        plt.draw()
        plt.axis("off")
        steps += 1
        # if steps == 1:
        #     plt.savefig("example_frame.png", dpi=300)
        plt.show()
        #print("action: ", act(obs)[0])
        episode_rew += rew
    print("Episode reward", episode_rew)


if __name__ == '__main__':
    LOAD_CKPT = "agents/29999_MLP-DQN"
    main(LOAD_CKPT)
