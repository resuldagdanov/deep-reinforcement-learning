from typing import Dict, Any
from copy import deepcopy
from collections import namedtuple
import numpy as np
import torch

from dqn.replaybuffer.uniform import UniformBuffer, BaseBuffer
from dqn.replaybuffer.prioritized import PriorityBuffer
from dqn.dqn.model import DQN


class RainBow(DQN):
    """
        Rainbow DQN agent with selectable extensions.

        Args:
            valuenet (torch.nn.Module): Q network
            nact (int): Number of actions
            extensions (Dict[str, Any]): Extension information
    """

    def __init__(self, valuenet: torch.nn.Module, nact: int, extensions: Dict[str, Any], *buffer_args):
        self.extensions = extensions
        
        if extensions["prioritized"]:
            buffer = PriorityBuffer(*buffer_args, alpha=extensions["prioritized"]["alpha"])
        else:
            buffer = UniformBuffer(*buffer_args)

        super().__init__(valuenet, nact, buffer)

    def greedy_policy(self, state: torch.Tensor, *args) -> int:
        """
            The greedy policy that changes its behavior depending on the value of the "distributional" option in the extensions dictionary.
            If distributional values are activated, use expected_value method.

            Args:
                state (torch.Tensor): Torch state

            Returns:
                int: action
        """

        if self.extensions["distributional"]:
            value_dist = self.valuenet(state)
            value_dist = value_dist.view(1, self.nact, -1)

            values = self.expected_value(value_dist)
            return values.sum(2).max(1)[1].detach()[0].item()
        
        else:
            return super().greedy_policy(state)

    def loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """
            Loss method that switches loss function depending on the value of the "distributional" option in extensions dictionary. 

            Args:
                batch (BaseBuffer.Transition): Batch of Torch Transitions
                gamma (float): Discount Factor

            Returns:
                torch.Tensor: Value loss
        """

        if self.extensions["distributional"]:
            return self.distributional_loss(batch, gamma)
        else:
            return self.vanilla_loss(batch, gamma)

    def vanilla_loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """
            MSE (L2, L1, or smooth L1) TD loss with double DQN extension in mind.
            Different than DQN loss, we keep the batch axis to make this compatible with the prioritized buffer.
            Note that: For target value calculation "_next_action_network" should be used.
            Set target network and action network to eval mode while calculating target value if the networks are noisy.

            Args:
                batch (BaseBuffer.Transition): Batch of Torch Transitions
                gamma (float): Discount Factor

            Returns:
                torch.Tensor: Value loss
        """
        
        # getting an available device (cuda is prefered)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # the size of the batch
        batch_size = batch.state.shape[0]

        # decode transition batch tuples and represent in torch tensor format
        state = torch.Tensor(batch.state).to(device)
        action = torch.Tensor(batch.action).to(device)
        next_state = torch.Tensor(batch.next_state).to(device)
        reward = torch.Tensor(batch.reward).to(device).view(batch_size)
        terminal = torch.Tensor(batch.terminal).float().to(device)

        multi_terminal = 1 - terminal.clone().view(batch_size)

        # determine which target or value network to use
        next_state_action = self._next_action_network(next_state).argmax(dim=1, keepdim=True)

        # target network (comes from BaseDQN class) is used to find the values of the next-states
        value_next_state = self.targetnet(next_state).gather(1, next_state_action).view(-1)
        value_next_state = value_next_state * multi_terminal

        # compute Bellman expectation
        expected_value_state = (value_next_state * gamma) + reward

        # get action state value from current value network
        value_state = self.valuenet(state).gather(1, action[:, 0].view(-1,1).long()).view(-1)

        # calculate loss between expected value and current value of the state
        loss_function = torch.nn.SmoothL1Loss(reduction="none")
        loss = loss_function(value_state, expected_value_state)

        return loss

    def expected_value(self, values: torch.Tensor) -> torch.Tensor:
        """
            Return the expected state-action values. Used when distributional value is activated.

            Args:
                values (torch.Tensor): Value tensor of distributional output (B, A, Z). B,
                    A, Z denote batch, action, and atom respectively.

            Returns:
                torch.Tensor: the expected value of shape (B, A)
        """
        
        v_min = self.extensions["distributional"]["vmin"]
        v_max = self.extensions["distributional"]["vmax"]
        n_atoms = self.extensions["distributional"]["natoms"]

        probabilities = torch.linspace(v_min, v_max, n_atoms).cuda()
        expected_next_value = values.mul(probabilities)

        return expected_next_value

    def distributional_loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """
            Distributional RL TD loss with KL divergence (with Double Q-learning via "_next_action_network" at target value calculation).
            Keep the batch axis. Set noisy network to evaluation mode while calculating target values.

            Args:
                batch (BaseBuffer.Transition): Batch of Torch Transitions
                gamma (float): Discount Factor

            Returns:
                torch.Tensor: Value loss
        """

        # getting an available device (cuda is prefered)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # the size of the batch
        batch_size = batch.state.shape[0]

        # decode transition batch tuples and represent in torch tensor format
        state = torch.Tensor(batch.state).to(device)
        action = torch.Tensor(batch.action).to(device)
        next_state = torch.Tensor(batch.next_state).to(device)
        reward = torch.Tensor(batch.reward).to(device).view(batch_size)
        terminal = torch.Tensor(batch.terminal).float().to(device)

        v_min = self.extensions["distributional"]["vmin"]
        v_max = self.extensions["distributional"]["vmax"]
        n_atoms = self.extensions["distributional"]["natoms"]

        delta_z = float(v_max - v_min) / (n_atoms - 1)
        probabilities = torch.linspace(v_min, v_max, n_atoms).cuda()

        # compute action distribution given the distributional probabilities
        next_action_distribution = self._next_action_network.forward(next_state).detach().mul(probabilities)

        # unsqueeze to make it compatible with the distributional loss
        next_action = next_action_distribution.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, n_atoms)

        # get target network value for the given batch of next states
        target_value = self.targetnet.forward(next_state).detach().mul(probabilities)

        # compute target action distribution
        target_next_distribution = target_value.gather(1, next_action).squeeze(1)

        # compute transition variables given next action distribution
        reward = reward.unsqueeze(1).expand_as(target_next_distribution)
        terminal = terminal.expand_as(target_next_distribution)
        probabilities = probabilities.unsqueeze(0).expand_as(target_next_distribution)

        # compute Bellman
        bellman = reward + (probabilities * gamma * (1 - terminal))
        bellman = bellman.clamp(min=v_min, max=v_max)

        b = (bellman - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * n_atoms, batch_size).long().unsqueeze(1).expand_as(target_next_distribution).cuda()

        proj_dist = torch.zeros_like(target_next_distribution, dtype=torch.float32).to(device)
        proj_dist.view(-1).index_add_(0, (offset + l).view(-1), (target_next_distribution * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (offset + u).view(-1), (target_next_distribution * (b - l.float())).view(-1))

        # compute distributional action from the current value network
        distribution = self.valuenet(state)
        
        action = action.unsqueeze(1).expand(batch_size, 1, n_atoms)
        action = action.type(torch.int64)

        distribution = distribution.gather(1, action).squeeze(1)
        distribution.detach().clamp_(min=1e-3)

        # compute distributional loss
        loss = - (proj_dist * distribution.log()).sum(1)

        return loss

    @property
    def _next_action_network(self) -> torch.nn.Module:
        """
            Return the network used for the next action calculation (Used for Double Q-learning)

            Returns:
                torch.nn.Module: Q network to find target/next action
        """
        
        if self.extensions["double"]:
            return self.valuenet
        else:
            return self.targetnet
