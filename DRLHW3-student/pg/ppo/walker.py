from typing import Tuple
import argparse
import numpy as np
import torch
import gym

from pg.ppo.model import PPO
from pg.a2c.vecenv import ParallelEnv


class DenseNet(torch.nn.Module):

    def __init__(self, in_size: int, out_size: int, hidden: int = 128):
        """
            Dense network that contains the value and policy functions.

            Args:
                in_size (int): Input size (length of the state vector)
                out_size (int): Action size (number of categories)
                hidden (int, optional): Hidden neuron size. Defaults to 128.
        """

        super().__init__()

        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_size)
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        """
            Return policy distribution and value

            Args:
                state (torch.Tensor): State tensor

            Returns:
                Tuple[torch.distributions.Normal, torch.Tensor]: Normal policy distribution and value
        """
        
        policy_logits = self.policy_net(state)
        policy = torch.distributions.Categorical(policy_logits)
        
        value = self.value_net(state)

        return policy, value


def make_env(envname: str) -> gym.Env:
    """
        Environment creating function
    """

    return gym.make(envname)


def main(args):
    """
        Start the learning process with the given arguments
    """

    seed = args.seed or np.random.randint(2**10, 2**30)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vecenv = ParallelEnv(args.nenv, lambda: make_env(args.envname), seed=seed)

    # we need to initialize an environment to get the dimensions
    env = make_env(args.envname)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]
    
    # we no longer need to keep this environment
    del env

    network = DenseNet(in_size, out_size, args.hidden_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    agent = PPO(network, args, vecenv, optimizer)
    agent.to(args.device)
    agent.learn()

    vecenv.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PPO with Bipedal Walker")
    parser.add_argument("--envname", type=str, default="BipedalWalker-v3", help="Name of the environment")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the experiment")
    parser.add_argument("--nenv", type=int, help="Number of environemnts run in parallel", default=16)
    parser.add_argument("--lr", type=float, help="Learning rate", default=3e-4)
    parser.add_argument("--device", type=str, help="Torch device", default="cpu")
    parser.add_argument("--timesteps", type=int, help="Number of timesteps to run learning", default=int(2e6))
    parser.add_argument("--n-step", type=int, help="Length of the rollout", default=2048)
    parser.add_argument("--hidden-size", type=int, help="Number of neurons in the hidden layers and gru", default=128)
    parser.add_argument("--gamma", type=float, help="Discount factor", default=0.99)
    parser.add_argument("--gae_lambda", type=float, help="lambda coefficient", default=0.95)
    parser.add_argument("--entropy_coef", type=float, help="Entropy coefficient", default=0.001)
    parser.add_argument("--value_coef", type=float, help="Value coefficient", default=0.5)
    parser.add_argument("--batch-size", type=int, help="Batch size of the parameter updates", default=64)
    parser.add_argument("--n_epochs", type=int, help="Number of epochs per rollout", default=10)
    parser.add_argument("--clip-range-max", type=float, help="Initial clip-range value", default=0.2)
    parser.add_argument("--clip-range-min", type=float, help="Minimum clip range value", default=0.2)
    parser.add_argument("--write-period", type=int, help="Logging period (interms of timesteps)", default=16 * 2048 * 1)
    parser.add_argument("--log-window-length", type=int, help="Last n episodic rewards to log", default=50)
    parser.add_argument("--log-dir", type=str, help="Logging directory", default=None)
    
    args = parser.parse_args()
    main(args)
