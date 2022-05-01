from typing import Generator
from collections import namedtuple, deque
from functools import reduce
import argparse
import gym
import torch

from .model import RainBow
from dqn.common import linear_annealing, exponential_annealing, PrintWriter
from dqn.dqn.train import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    """
        Training class that organize evaluation, update, and transition gathering.
            Arguments:
                - args: Parser arguments
                - agent: RL agent object
                - opt: Optimizer that optimizes agent's parameters
                - env: Gym environment
    """

    def __init__(self, args: argparse.Namespace, agent: RainBow, opt: torch.optim.Optimizer, env: gym.Env):
        """
            Training class that organize evaluation, update, and transition gathering.

            Args:
                args (argparse.Namespace): CL arguments
                agent (RainBow): RainBow Agent
                opt (torch.optim.Optimizer): Optimizer for agents parameters
                env (gym.Env): Environment
        """

        super().__init__(args, agent, opt, env)

        # beta = 1 - prioritized_beta
        self.prioritized_beta = linear_annealing(
            init_value=1 - args.beta_init,
            min_value=0,
            decay_range=args.n_iterations
        )

    def update(self, iteration: int) -> None:
        """
            One step updating function. Update the agent in training mode.
            - clip gradient if "clip_grad" is given in args.
            - keep track of td loss. Append td loss to "self.td_loss" list
            - Update target network.

            If the prioritized buffer is active:
                - Use the weighted average of the loss where the weights are returned by the prioritized buffer
                - Update priorities of the sampled transitions

            If noisy-net is active:
                - reset noise for valuenet and targetnet
            Check for the training index "iteration" to start the update.

            Args:
                iteration (int): Training iteration
        """
        
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        return super().update(iteration)

    def __iter__(self) -> Generator[RainBow.Transition, None, None]:
        """
            n-step transition generator. Yield a transition with n-step look ahead. Use the greedy policy if noisy network extension is activate.

            Yields:
                Generator[RainBow.Transition, None, None]: Transition of (s_t, a_t, \sum_{j=t}^{t+n}(\gamma^{j-t} r_j), done, s_{t+n})
        """
        
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        yield from super().__iter__()
