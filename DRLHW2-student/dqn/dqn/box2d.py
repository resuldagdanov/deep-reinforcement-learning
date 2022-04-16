import torch
import gym
import argparse
import numpy as np
import random

from dqn.dqn.model import DQN
from dqn.replaybuffer.uniform import UniformBuffer
from dqn.dqn.train import Trainer


class ValueNet(torch.nn.Module):
    """ Fully connected neural network to estimate Q values. 

    Args:
        in_size (int): State feature size
        out_size (int): Number of action
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Run the value network with the given state

        Args:
            state (torch.Tensor): Torch state

        Returns:
            torch.Tensor: Q Value outputs
        """
        raise NotImplementedError
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/


def main(args):
    """ The main function that prepares a model and starts training  """
    env = gym.make(args.envname)
    env._max_episode_steps = args.max_episode_len
    state_shape = env.observation_space.shape
    state_dtype = env.observation_space.dtype
    act_size = env.action_space.n

    seed = np.random.randint(2**1, 2**20) if args.seed is None else args.seed
    env.seed(seed)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    valuenet = ValueNet(state_shape[0], act_size)
    buffer = UniformBuffer(args.buffer_capacity, state_shape, state_dtype)
    agent = DQN(valuenet, act_size, buffer)
    optimizer = torch.optim.Adam(valuenet.parameters(), lr=args.lr)
    Trainer(args, agent, optimizer, env)()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument("--envname", type=str,
                        default="LunarLander-v2",
                        help="Name of the environment")
    parser.add_argument("--n-iterations", type=int, default=40000,
                        help="Number of training iterations")
    parser.add_argument("--start-update", type=int, default=100,
                        help="Number of iterations to wait before starting to update")
    parser.add_argument("--max-episode-len", type=int, default=300,
                        help="Maximum length of an episode before termination")

    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size of each update in training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount Factor")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning Rate")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--target-update-period", type=int, default=300,
                        help="Target network update period")
    parser.add_argument("--buffer-capacity", type=int, default=10000,
                        help="Replay buffer capacity")
    parser.add_argument("--epsilon-init", type=float, default=0.9,
                        help="Initial value of the epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.1,
                        help="Minimum value of the epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=None,
                        help="Epsilon decay rate for exponential decaying")
    parser.add_argument("--epsilon-range", type=int, default=None,
                        help="Epsilon decaying range for linear decay")
    parser.add_argument("--clip-grad", action="store_true",
                        help="Gradient Clip between -1 and 1. Default: No")

    parser.add_argument("--eval-period", type=int, default=500,
                        help="Evaluation period in terms of iteration")
    parser.add_argument("--eval-episode", type=int, default=5,
                        help="Number of episodes to evaluate the policy at evaluation")
    parser.add_argument("--save-model", action="store_true",
                        help="If given most successful models so far will be saved")
    parser.add_argument("--model-dir", type=str, default="models/",
                        help="Directory to save models")
    parser.add_argument("--write-period", type=int, default=100,
                        help="Logging period in terms of iterations")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Logging directory. Default: /tmp")
    parser.add_argument("--render", action="store_false",
                        help="Render evaluations")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed value. Default: Random seed")

    args = parser.parse_args()
    main(args)
