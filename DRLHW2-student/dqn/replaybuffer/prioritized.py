from lib2to3.pytree import Base
from typing import Tuple
import numpy as np

from dqn.replaybuffer.uniform import BaseBuffer


class PriorityBuffer(BaseBuffer):
    """
        Prioritized Replay Buffer. 

        Args:
            capacity (int): Maximum size of the buffer
            state_shape (Tuple): Shape of a single the state
            state_dtype (np.dtype): Data type of the states
            alpha (float): Exponent of td
            epsilon (float, optional): Minimum absolute td error. If |td| = 0.2, we store it as |td| + epsilon. Defaults to 0.1.
    """

    def __init__(self, capacity: int, state_shape: Tuple[int], state_dtype: np.dtype, alpha: float, epsilon: float = 0.1):
        super().__init__(capacity, state_shape, state_dtype)

        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha

        # pointing to the next writing index
        self.write_index = 0
        self.size = 0

        # maximum = epsilon at the beginning
        self.max_abs_td = epsilon

        # initialize list of priorities with zeros
        self.abs_td_errors = np.zeros((capacity,), dtype=np.float32) + 1e-6

    def push(self, transition: BaseBuffer.Transition) -> None:
        """
            Push a transition object (with single elements) to the buffer.
            Transitions are pushed with the current maximum absolute td.
            Remember to set <write_index> and <size> attributes.

            Args:
                transition (BaseBuffer.Transition): transition to push to buffer
        """
        
        # extract the elements from the transition
        current_state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        terminal = transition.terminal

        # store the transition elements in the buffer with corresponding write_index
        self.buffer.state[self.write_index] = current_state
        self.buffer.action[self.write_index] = action
        self.buffer.reward[self.write_index] = reward
        self.buffer.next_state[self.write_index] = next_state
        self.buffer.terminal[self.write_index] = terminal

        # update the priority of the write_index
        max_priority = self.abs_td_errors.max() if self.buffer else 1.0
        self.abs_td_errors[self.write_index] = max_priority

        # update the write_index
        self.write_index = (self.write_index + 1) % self.capacity

        # update the size of the buffer, remember that the buffer size could not exeed the capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Tuple[BaseBuffer.Transition, np.ndarray, np.ndarray]:
        """
            Sample a batch of transitions based on priorities.

            Args:
                batch_size (int): Batch size
                beta (float): Exponent of IS weights

            Returns:
                Tuple[BaseBuffer.Transition, np.ndarray, np.ndarray]:
                    - Transition object of batch of samples
                    - Indices of the samples (used for priority update)
                    - Importance sampling weights
        """

        # do not sample if the buffer size is less than the batch size
        if batch_size > self.size:
            return None

        # get priority values from the buffer
        if self.size < self.capacity:
            all_priorities = self.abs_td_errors[:self.write_index]
        else:
            all_priorities = self.abs_td_errors
        
        # define probabilities
        probability_alpha = np.array(all_priorities, dtype=np.float32) ** self.alpha
        probabilities = probability_alpha / probability_alpha.sum()

        # sample random indices of the transitions of size batchsize and sort them
        # note that we use the module operator to ensure that the indices are within the buffer size
        # samples are not replaced, meaning that the value a could not be sampled twice in one batch
        random_index = np.random.choice(self.size, size=batch_size, p=probabilities, replace=False)
        random_index = np.sort(random_index)

        # extract prioritized experience samples from the buffer of size batchsize
        current_state = self.buffer.state[random_index]
        action = self.buffer.action[random_index]
        reward = self.buffer.reward[random_index]
        next_state = self.buffer.next_state[random_index]
        terminal = self.buffer.terminal[random_index]

        # store the sampled transition in a namedtuple format
        transition = self.Transition(current_state, action, reward, next_state, terminal)

        # compute importance sampling weights
        weights_beta = (self.size * probabilities[random_index]) ** (-beta)
        weights = weights_beta / weights_beta.max()

        return transition, random_index, weights

    def update_priority(self, indices: np.ndarray, td_values: np.ndarray) -> None:
        """
            Update the priority td_values of given indices (returned from sample).
            Update max_abs_td value if there exists a higher absolute td.

            Args:
                indices (np.ndarray): Indices of the samples
                td_values (np.ndarray): New td values
        """
        
        # update with higher absolute td
        self.max_abs_td = max(abs(td_values))

        # update priority of each index sample
        for index, value in zip(indices, td_values):
            self.abs_td_errors[index] = value
