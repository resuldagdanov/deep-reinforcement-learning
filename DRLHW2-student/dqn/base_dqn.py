from typing import Tuple
import torch
import gym
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod

from dqn.replaybuffer.uniform import BaseBuffer


class BaseDQN(torch.nn.Module, ABC):
    """ Base Class for DQN and agents.

        Args:
            - valuenet (torch.nn.Module): Neural network to estimate values
            - nact (int): Number of actions (or outputs)
    """

    Transition = BaseBuffer.Transition

    def __init__(self, valuenet: torch.nn.Module, nact: int):
        super().__init__()
        self.valuenet = valuenet
        self.nact = nact
        self.targetnet = deepcopy(valuenet)

    def greedy_policy(self, state: torch.Tensor) -> int:
        """ Find the action that has the highest value for the given state

        Args:
            state (torch.Tensor): Environment state of shape (1, D)

        Returns:
            int: selected action
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

    def e_greedy_policy(self, state: torch.Tensor, epsilon: float) -> int:
        """ Randomly (Bernoulli distribution with p equals to epsilon) select an 
        action (ranging from 0 to nact-1) or select the greedy action.

        Args:
            state (torch.Tensor): _description_
            epsilon (float): _description_

        Returns:
            int: action
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.nact)
        else:
            return self.greedy_policy(state)

    @abstractmethod
    def push_transition(self, transition: BaseBuffer.Transition) -> None:
        """ Push transition to Replay Buffer

        Args:
            transition (Transition): Transition namedtuple
        """
        pass

    @abstractmethod
    def loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """ TD loss function

        Args:
            batch (Transition): Batch of transitions
            gamma (float): Discount factor

        Returns:
            torch.Tensor: TD loss
        """
        pass

    def update_target(self) -> None:
        """ Update the target network by setting its parameters to valuenet
        parameters """
        raise NotImplementedError
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/

    def evaluate(self, eval_episode: int, env: gym.Env, device: str, render: bool = False) -> float:
        """ Agent evaluation function. Evaluate the current greedy policy for
        eval_episode many "full" episodes. Return the mean episodic reward
        (average of total rewards per episode).

        Args:
            eval_episode (int): Number of episodes to evaluate
            env (gym.Env): Environmnet
            device (str): Torch device
            render (bool): If true render the environment

        Returns:
            float: Average episodic reward
        """
        self.eval()
        raise NotImplementedError
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/

    @staticmethod
    def batch_to_torch(batch: BaseBuffer.Transition, device: str) -> BaseBuffer.Transition:
        """ Convert numpy transition into a torch transition
        Note: Dtype of actions is "long" while the remaining dtypes are
        "float32"

        Args:
            batch (Transition): Batch of Numpy transitions
            device (str): Torch device

        Returns:
            Transition: Batch of Torch transitions
        """
        return BaseDQN.Transition(
            *(torch.from_numpy(x).type(dtype).to(device)
              for x, dtype in zip(
                batch,
                (torch.float,
                 torch.long,
                 torch.float32,
                 torch.float32,
                 torch.float32)))
        )

    @staticmethod
    def state_to_torch(state: np.ndarray, device: str) -> torch.Tensor:
        """ Convert numpy state into torch state

        Args:
            state (np.ndarray): Numpy environment state
            device (str): Torch device

        Returns:
            torch.Tensor: Torch environment state
        """
        return torch.from_numpy(state).float().to(device)
