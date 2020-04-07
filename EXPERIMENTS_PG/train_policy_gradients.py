import argparse
import torch

from drl_toolbox.utils import DotDic
from drl_toolbox.utils import set_random_seeds, load_config
from drl_toolbox.dl import BodyBuilder, RecurrentBodyBuilder
from drl_toolbox.dl import set_optimizer

from drl_toolbox.single_rl import make_parallel_env
from drl_toolbox.single_rl.agents import VPG_Agent


def main(train_config, net_config, log_config):
    """ Train & evaluate a policy gradient agent. """
    # Set the training device & the random seed for the example run
    device_name = train_config.device_name
    device = torch.device(device_name)
    set_random_seeds(seed_id=train_config.seed_id, verbose=False)

    # Define the DQN network architecture & optimizer
    policy_network = BodyBuilder(**net_config).to(device)
    optimizer = set_optimizer(network=policy_network,
                              opt_type=train_config.opt_type,
                              l_rate=train_config.l_rate)

    # Agent - Envs, Nets, Optimizer, Train Config, Logger
    train_log = DeepLogger(**log_config)
    train_env = make_parallel_env(train_config.env_name,
                                  train_config.num_train_threads)
    test_env = make_parallel_env(train_config.env_name,
                                 train_config.num_test_threads)

    pg_agent = VPG_Agent(train_env, test_env, train_config, train_log,
                         policy_network, optimizer, device)

    # Run the training of the DQN agent
    pg_agent.run_learning_loop(num_train_updates=train_config.num_train_updates)
    return train_log, policy_network


if __name__ == "__main__":
    def get_cmd_args():
        """ Get env name, config file path & device to train from cmd line """
        parser = argparse.ArgumentParser()
        parser.add_argument('-config', '--config_fname', action="store",
                            default="configs/base_vpg.json", type=str,
                            help='Filename from which to load config')
        return parser.parse_args()

    cmd_args = get_cmd_args()
    config = load_config(cmd_args.config_fname)
    # config.log_config.config_fname = cmd_args.config_fname

    net_config = DotDic(config["net_config"])
    train_config = DotDic(config["train_config"])
    log_config = dict(config["log_config"])
    main(train_config, net_config, log_config)
