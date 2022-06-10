import argparse
import torch
import numpy as np
import gym

from model import Reinforce


class PolicyNet(torch.nn.Module):

    def __init__(self, in_size: int, act_size: int, hidden_size: int = 128):
        """
            Simple policy network that returns a Categorical distribution.

            Args:
                in_size (int): Size of the input
                act_size (int): Action category size
                hidden_size (int, optional): Hidden neuron size. Defaults to 128.
        """

        super().__init__()
        
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

    def forward(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """
            Return pi(a|s) in the form of a Categorical distribution

            Args:
                state (torch.Tensor): State tensor

            Returns:
                torch.distributions.Categorical: Categorical policy distribution for the given state
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError


def main(args):
    """
        Start the learning process with the given arguments (hyperparameters)
    """

    env = gym.make(args.envname)

    seed = args.seed or np.random.randint(2**10, 2**30)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    in_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    policynet = PolicyNet(in_size,
                          act_size,
                          hidden_size=args.hidden_size)
    agent = Reinforce(policynet, log_dir=args.log_dir)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)

    agent.to(args.device)
    agent.learn(args, opt, env)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="REINFORCE in Cartpole")
    parser.add_argument("--envname", type=str, default="CartPole-v1", help="Name of the environment")
    parser.add_argument("--n-episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--max-episode-len", type=int, default=1000, help="Maximum length of an episode before termination")
    parser.add_argument("--write-period", type=int, default=20, help="Logging period in terms of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the experiment")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount Factor")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning Rate")
    parser.add_argument("--hidden-size", type=int, default=128, help="Number of neurons in the hidden layers of the policy network")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--clip-grad", action="store_true", help="Gradient Clip between -1 and 1. Default: No")
    parser.add_argument("--log-window-length", type=int, help="Last n episodic rewards to log", default=20)
    parser.add_argument("--log-dir", type=str, help="Logging directory", default=None)

    args = parser.parse_args()
    main(args)
