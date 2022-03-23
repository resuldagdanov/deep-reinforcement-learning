""" Function apporximation methods in TD
    Author: Your-name / Your-number
"""
import numpy as np
import time
from collections import namedtuple


class ApproximateAgent():
    r""" Base class for the approximate methods. This class
    provides policies and training loop. Initiate a weight matrix
    with shape (#observation x #action).
    """

    def __init__(self, nobs, nact):
        self.nact = nact
        self.nobs = nobs
        self.weights = np.random.uniform(-0.1, 0.1, size=(nobs, nact))

    def q_values(self, state):
        """ Return the q values of the given state for each action.
        """
        return np.dot(state, self.weights)

    def greedy_policy(self, state, *args):
        """ Return the best possible action according to the value
        function """
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|

    def e_greedy_policy(self, state, epsilon):
        """ Policy that returns the best action according to q values with
        (epsilon/#action) + (1 - epsilon) probability and any other action with
        probability episolon/#action.
        """
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|

    def train(self, env, policy, args):
        """ Training loop for the approximate agents.
        Initiate an episodic reward list and a loss list. At each episode
        decrease the epsilon value exponentially using args.eps_decay_rate
        within the boundries of the args.init_eps and args.final_eps. At each
        transition update the agent(weights of the function). For every
        "args._evaluate_period"'th step call evaluation function and store the
        returned episodic reward to the reward list.

        Arguments:
            - env: gym environment
            - policy: Behaviour policy to be used in training(not in
            evaluation)
            - args: namedtuple of hyperparameters

        Return:
            - Episodic reward list of evaluations (not the training rewards)
            - Loss list of the training (one loss for per update)
        """
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|

    def update(self, *arg, **kwargs):
        raise NotImplementedError

    def evaluate(self, env):
        raise NotImplementedError


class ApproximateQAgent(ApproximateAgent):
    r""" Approximate Q learning agent where the learning is done
    via minimizing the mean squared value error with semi-gradient descent.
    This is an off-policy algorithm.
    """

    def __init__(self, nobs, nact):
        super().__init__(nobs, nact)

    def update(self, transition, alpha, gamma):
        """ Update the parameters that parameterized the value function
        according to (semi-gradient) q learning.

        Arguments:
            - transition: 4 tuple of state, action, reward and next_state
            - alpha: Learning rate of the update function
            - gamma: Discount rate

        Return:
            Mean squared temporal difference error
        """
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|

    def evaluate(self, env, render=False):
        """ Single episode evaluation of the greedy agent.
        Arguments:
            - env: gym environemnt
            - render: If true render the environment(default False)
        Return:
            Episodic reward
        """
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|
