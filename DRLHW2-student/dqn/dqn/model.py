import torch
import numpy as np

from dqn.replaybuffer.uniform import UniformBuffer
from dqn.base_dqn import BaseDQN


class DQN(BaseDQN):
    """ Deep Q Network agent.

    Args:
        valuenet (torch.nn.Module): Neural network to estimate Q values
        nact (int):  Number of actions (or outputs)
        buffer (UniformBuffer): Uniform Replay Buffer
    """

    def __init__(self, valuenet: torch.nn.Module, nact: int, buffer: UniformBuffer):
        super().__init__(valuenet, nact)
        self.buffer = buffer

    def push_transition(self, transition: UniformBuffer.Transition) -> None:
        """ Push transition to replay buffer.

        Args:
            transition (UniformBuffer.Transition): One step transition
        """
        self.buffer.push(transition)

    def loss(self, batch: UniformBuffer.Transition, gamma: float) -> torch.Tensor:
        """ TD loss that uses the target network to estimate target values

        Args:
            batch (UniformBuffer.Transition): Batch of transitions
            gamma (float): Discount factor

        Returns:
            torch.Tensor: TD loss
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
