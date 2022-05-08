import torch

from dqn.replaybuffer.uniform import UniformBuffer
from dqn.base_dqn import BaseDQN


class DQN(BaseDQN):
    """
        Deep Q Network agent.

        Args:
            valuenet (torch.nn.Module): Neural network to estimate Q values
            nact (int):  Number of actions (or outputs)
            buffer (UniformBuffer): Uniform Replay Buffer
    """

    def __init__(self, valuenet: torch.nn.Module, nact: int, buffer: UniformBuffer):
        super().__init__(valuenet, nact)
        self.buffer = buffer

    def push_transition(self, transition: UniformBuffer.Transition) -> None:
        """
            Push transition to replay buffer.

            Args:
                transition (UniformBuffer.Transition): One step transition
        """
        
        self.buffer.push(transition)

    def loss(self, batch: UniformBuffer.Transition, gamma: float) -> torch.Tensor:
        """
            TD loss that uses the target network to estimate target values

            Args:
                batch (UniformBuffer.Transition): Batch of transitions
                gamma (float): Discount factor

            Returns:
                torch.Tensor: TD loss
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

        # get action state value from current value network
        value_state = self.valuenet(state).gather(1, action[:, 0].view(-1, 1).long()).view(-1)

        # target network (comes from BaseDQN class) is used to find the values of the next-states
        value_next_state, _ = torch.max(self.targetnet(next_state).to(device), dim=1)

        # compute Bellman expectation
        expected_value_state = reward + (value_next_state * gamma * (1 - terminal.view(batch_size)))

        # calculate loss between expected value and current value of the state
        loss_function = torch.nn.MSELoss(reduction='mean')
        loss = loss_function(value_state, expected_value_state)

        return loss
