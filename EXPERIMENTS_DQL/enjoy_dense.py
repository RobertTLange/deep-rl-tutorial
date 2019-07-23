import gym
import gridworld
from dqn import init_dqn, MLP_DQN
from pycolab import rendering
import torch

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import glob
import os
import subprocess
import argparse


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


def main(LOAD_CKPT, save_fname, title):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

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
    while steps < 50 and not done:
        #obs, rew, done, _ , screen_obs = env.step_with_render(act(obs)[0])
        #obs, rew, done, _ , screen_obs = env.step_with_render(env.action_space.sample())
        action = agents["current"].act(obs.flatten(), epsilon=0.05)
        obs, rew, done, _ , screen_obs = env.step_with_render(action)
        converted = converter(screen_obs)
        plt.ion()
        my_plot.autoscale()
        my_plot.set_data(converted)
        plt.title(title + "- Step: {}".format(steps + 1))
        plt.pause(.05)
        plt.draw()
        plt.axis("off")
        steps += 1
        # if steps == 1:
        #     plt.savefig("example_frame.png", dpi=300)
        plt.savefig("movies/file%02d.png" % steps)
        #print("action: ", act(obs)[0])
        episode_rew += rew
    print("Episode reward", episode_rew)

    os.chdir(os.getcwd() + "/movies")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        save_fname
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General logging/saving and device arguments
    parser.add_argument('-agent_fname', '--AGENT', action="store",
                        default="500000_MLP-DQN", type=str,
                        help='Filename of DQN agent.')
    parser.add_argument('-title', '--TITLE', action="store",
                        default="500000", type=str,
                        help='Iteration Title on top of frame.')
    args = parser.parse_args()


    LOAD_CKPT = "agents/" + args.AGENT
    save_fname = args.AGENT + ".mp4"
    title = "DQN Agent after {} Iterations".format(args.TITLE)
    main(LOAD_CKPT, save_fname, title)
