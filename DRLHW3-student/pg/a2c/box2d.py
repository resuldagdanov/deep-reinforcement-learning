from typing import Tuple
import argparse
import numpy as np
import torch
import gym

from pg.a2c.model import A2C
from pg.a2c.vecenv import ParallelEnv


class GruNet(torch.nn.Module):
    """
        Gru network that combines value and policy functions.

        Args:
            in_size (int): Input size (length of the state vector)
            out_size (int): Action size (number of categories)
            hidden (int, optional): Hidden neuron size. Defaults to 128.
    """

    def __init__(self, in_size: int, out_size: int, hidden: int = 128):
        super().__init__()
        n_layers = 1
        
        self.gru = torch.nn.GRU(input_size=in_size, hidden_size=hidden, num_layers=n_layers, batch_first=True)
        
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_size)
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, state: torch.Tensor, gru_hx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Return policy logits, value, and the gru hidden state for the given state

            Args:
                state (torch.Tensor): State tensor
                gru_hx (torch.Tensor): gru hidden state

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: policy logits, value, and gru hidden state
        """

        gru_logits, gru_hx = self.gru(state, gru_hx)

        policy_logits = self.policy_net(gru_logits)
        value = self.value_net(gru_logits)

        return policy_logits, value, gru_hx


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
    out_size = env.action_space.n
    
    # we no longer need to keep this environment
    del env

    network = GruNet(in_size, out_size, args.hidden_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    agent = A2C(network, args, vecenv, optimizer)
    agent.to(args.device)
    agent.learn()

    vecenv.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A2C with Box2d")
    parser.add_argument("--envname", type=str, default="LunarLander-v2", help="Name of the environment")
    parser.add_argument("--nenv", type=int, help="Number of environments run in parallel", default=16)
    parser.add_argument("--seed", type=int, default=None, help="Seed of the experiment")
    parser.add_argument("--lr", type=float, help="Learning rate", default=3e-4)
    parser.add_argument("--device", type=str, help="Torch device", default="cpu")
    parser.add_argument("--timesteps", type=int, help="Number of timesteps to run learning", default=int(2e6))
    parser.add_argument("--n-step", type=int, help="Length of the rollout", default=5)
    parser.add_argument("--hidden-size", type=int, help="Number of neurons in the hidden layers and gru", default=128)
    parser.add_argument("--gamma", type=float, help="Discount factor", default=0.995)
    parser.add_argument("--gae_lambda", type=float, help="lambda coefficient", default=1.0)
    parser.add_argument("--entropy_coef", type=float, help="Entropy coefficient", default=0.1)
    parser.add_argument("--value_coef", type=float, help="Value coefficient", default=0.5)
    parser.add_argument("--write-period", type=int, help="Logging period (in terms of timesteps)", default=16 * 5 * 100)
    parser.add_argument("--log-window-length", type=int, help="Last n episodic rewards to log", default=50)
    parser.add_argument("--log-dir", type=str, help="Logging directory", default=None)

    args = parser.parse_args()
    main(args)
