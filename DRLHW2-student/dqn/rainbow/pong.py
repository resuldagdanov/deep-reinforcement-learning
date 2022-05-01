from typing import Dict, Any
import torch
import numpy as np
import gym
import argparse
import random

from dqn.rainbow.model import RainBow
from dqn.rainbow.train import Trainer
from dqn.rainbow.layers import HeadLayer, NoisyLinear
from dqn.common import ResizeAndScalePong, NoopResetEnv, DerivativeEnv, DoubleActionPong


class ValueNet(torch.nn.Module):
    """
        Convolutional neural network to estimate Q values. Use NoisyLinear instead of Linear ones if noisy is True in the extensions.
        Final layer must be HeadLayer.

        Args:
            in_size (int): State feature size
            out_size (int): Number of action
            extensions (Dict[str, Any]): A dictionary that keeps extension information
    """

    def __init__(self, in_channel: int, out_size: int, extensions: Dict[str, Any]):
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
        """
            Run the value network with the given state

            Args:
                state (torch.Tensor): Torch state of shape (B, C, H, W)

            Returns:
                torch.Tensor: Q Value outputs
        """

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/

    def reset_noise(self) -> None:
        """
            Call reset_noise function of all child layers. Only used when "noisy" is active.
        """

        for module in self.children():
            if isinstance(module, (NoisyLinear, HeadLayer)):
                module.reset_noise()


def make_env():
    """
        Wrapped pong environment
    """
    
    env = gym.make("ALE/Pong-v5")
    env = ResizeAndScalePong(env)
    env = DerivativeEnv(env)
    env = NoopResetEnv(env, 20, 20)
    env = DoubleActionPong(env)
    
    return env


def main(args: argparse.Namespace) -> None:
    """
        The main function that prepares a model, extension dictionary and starts training

        Args:
            args (argparse.Namespace): CL arguments
    """

    env = make_env()
    state_shape = env.observation_space.shape
    state_dtype = env.observation_space.dtype
    act_size = env.action_space.n

    seed = np.random.randint(2**1, 2**20) if args.seed is None else args.seed
    env.seed(seed)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    extensions = {
        "double": not args.no_double,
        "dueling": not args.no_dueling,
        "noisy": False if args.no_noisy else {
            "init_std": args.noisy_std
        },
        "nstep": args.n_steps,
        "distributional": False if args.no_dist else {
            "vmin": args.vmin,
            "vmax": args.vmax,
            "natoms": args.natoms,
        },
        "prioritized": False if args.no_prioritized else {
            "alpha": args.alpha,
            "beta_init": args.beta_init
        },
    }

    valuenet = ValueNet(state_shape[0], act_size, extensions)

    agent = RainBow(valuenet, act_size, extensions, args.buffer_capacity, state_shape, state_dtype)
    
    optimizer = torch.optim.Adam(valuenet.parameters(), lr=args.lr)
    agent.to(args.device)
    
    Trainer(args, agent, optimizer, env)()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument("--n-iterations", type=int, default=1500000, help="Number of training iterations")
    parser.add_argument("--start-update", type=int, default=25000, help="Number of iterations until starting to update")

    # ----------------------- Hyperparameters -----------------------
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size of each update in training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount Factor")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--target-update-period", type=int, default=2500, help="Target network updating period")
    parser.add_argument("--buffer-capacity", type=int, default=250000, help="Replay buffer capacity")
    parser.add_argument("--epsilon-init", type=float, default=0.9, help="Initial value of the epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.1, help="Minimum value of the epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=None, help="Epsilon decay rate for exponential decaying")
    parser.add_argument("--epsilon-range", type=int, default=750000, help="Epsilon decaying range for linear decay")
    parser.add_argument("--clip-grad", action="store_true", help="Gradient Clip between -1 and 1. Default: No")

    # ----------------------- Extensions -----------------------
    parser.add_argument("--no-double", action="store_true", help="Disable double DQN extension")
    parser.add_argument("--no-dueling", action="store_true", help="Disable dueling DQN extension")
    parser.add_argument("--no-noisy", action="store_true", help="Disable noisy layers")
    parser.add_argument("--noisy-std", type=float, default=0.2,help="Initial std for noisy layers")
    parser.add_argument("--no-prioritized", action="store_true", help="Disable prioritized replay buffer")
    parser.add_argument("--alpha", type=float, default=0.6, help="Prioritization exponent")
    parser.add_argument("--beta-init", type=float, default=0.4, help="Prioritization exponent")
    parser.add_argument("--n-steps", type=int, default=1, help="Number of steps for bootstrapping")
    parser.add_argument("--no-dist", action="store_true", help="Disable distributional DQN extension")
    parser.add_argument("--vmin", type=float, default=-21, help="Minimum value for distributional DQN extension")
    parser.add_argument("--vmax", type=float, default=21, help="Maximum value for distributional DQN extension")
    parser.add_argument("--natoms", type=int, default=51, help="Number of atoms in distributional DQN extension")

    # ----------------------- Miscelenious -----------------------
    parser.add_argument("--eval-period", type=int, default=25000, help="Evaluation period in terms of iteration")
    parser.add_argument("--eval-episode", type=int, default=3, help="Number of episodes to evaluate")
    parser.add_argument("--save-model", action="store_true", help="If given most successful models so far will be saved")
    parser.add_argument("--model-dir", type=str, default="models/", help="Directory to save models")
    parser.add_argument("--write-period", type=int, default=2500, help="Writer period")
    parser.add_argument("--log_dir", type=str, default=None, help="Logging directory. Default: /tmp")
    parser.add_argument("--render", action="store_false", help="Render evaluations")
    parser.add_argument("--seed", type=int, default=None, help="Seed value. Default: Random seed")

    args = parser.parse_args()
    main(args)
