from typing import Dict, Any
from copy import deepcopy
from collections import namedtuple
import numpy as np
import torch

from dqn.replaybuffer.uniform import UniformBuffer, BaseBuffer
from dqn.replaybuffer.prioritized import PriorityBuffer
from dqn.dqn.model import DQN


class RainBow(DQN):
    """ Rainbow DQN agent with selectable extensions.

    Args:
        valuenet (torch.nn.Module): Q network
        nact (int): Number of actions
        extensions (Dict[str, Any]): Extension information
    """

    def __init__(self, valuenet: torch.nn.Module, nact: int, extensions: Dict[str, Any], *buffer_args):

        self.extensions = extensions
        if extensions["prioritized"]:
            buffer = PriorityBuffer(
                *buffer_args,
                alpha=extensions["prioritized"]["alpha"]
            )
        else:
            buffer = UniformBuffer(*buffer_args)
        super().__init__(valuenet, nact, buffer)

    def greedy_policy(self, state: torch.Tensor, *args) -> int:
        """ The greedy policy that changes its behavior depending on the
        value of the "distributional" option in the extensions dictionary. If
        distributional values are activated, use expected_value method.

        Args:
            state (torch.Tensor): Torch state

        Returns:
            int: action
        """
        if self.extensions["distributional"]:
            value_dist = self.valuenet(state)
            return self.expected_value(value_dist).argmax().item()
        return super().greedy_policy(state)

    def loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """ Loss method that switches loss function depending on the value
        of the "distributional" option in extensions dictionary. 

        Args:
            batch (BaseBuffer.Transition): Batch of Torch Transitions
            gamma (float): Discount Factor

        Returns:
            torch.Tensor: Value loss
        """
        if self.extensions["distributional"]:
            return self.distributional_loss(batch, gamma)
        return self.vanilla_loss(batch, gamma)

    def vanilla_loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """ MSE (L2, L1, or smooth L1) TD loss with double DQN extension in
        mind. Different than DQN loss, we keep the batch axis to make this
        compatible with the prioritized buffer. Note that: For target value calculation 
        "_next_action_network" should be used. Set target network and action network to
        eval mode while calculating target value if the networks are noisy.

        Args:
            batch (BaseBuffer.Transition): Batch of Torch Transitions
            gamma (float): Discount Factor

        Returns:
            torch.Tensor: Value loss
        """
        return super().loss(batch, gamma)
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/


    def expected_value(self, values: torch.Tensor) -> torch.Tensor:
        """ Return the expected state-action values. Used when distributional
            value is activated.

        Args:
            values (torch.Tensor): Value tensor of distributional output (B, A, Z). B,
                A, Z denote batch, action, and atom respectively.

        Returns:
            torch.Tensor: the expected value of shape (B, A)
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


    def distributional_loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """ Distributional RL TD loss with KL divergence (with Double
        Q-learning via "_next_action_network" at target value calculation).
        Keep the batch axis. Set noisy network to evaluation mode while calculating
        target values.

        Args:
            batch (BaseBuffer.Transition): Batch of Torch Transitions
            gamma (float): Discount Factor

        Returns:
            torch.Tensor: Value loss
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

    @property
    def _next_action_network(self) -> torch.nn.Module:
        """ Return the network used for the next action calculation (Used for
        Double Q-learning)

        Returns:
            torch.nn.Module: Q network to find target/next action
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

