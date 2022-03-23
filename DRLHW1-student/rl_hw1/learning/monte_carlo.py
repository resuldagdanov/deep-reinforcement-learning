""" Tabular MC algorithms
    Author: Your-name / Your-number
"""
from collections import defaultdict
import random
import math
from collections import deque
import numpy as np
import time


class TabularAgent:
    """ Based Tabular Agent class that inludes policies and evaluation function
    """

    def __init__(self, nact):
        self.qvalues = defaultdict(lambda: [0.0]*nact)
        self.nact = nact

    def greedy_policy(self, state, *args, **kwargs):
        """ Policy that returns the best action according to q values.
        """
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|

    def e_greedy_policy(self, state, epsilon, *args, **kwargs):
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

    def evaluate(self, env, render=False):
        """ Single episode evaluation of the greedy agent.
        Arguments:
            - env: Warehouse or Mazeworld environemnt
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


class MonteCarloAgent(TabularAgent):
    """ Tabular Monte Carlo Agent that updates q values based on MC method.
    """

    def __init__(self, nact):
        super().__init__(nact)

    def one_epsiode_train(self, env, policy, gamma, alpha):
        """ Single episode training function.
        Arguments:
            - env: Mazeworld environment
            - policy: Behaviour policy for the training loop
            - gamma: Discount factor
            - alpha: Exponential decay rate of updates

        Returns:
            episodic reward

        **Note** that in the book (Sutton & Barto), they directly assign the
        return to q value. You can either implmenet that algorithm (given in
        chapter 5) or use exponential decaying update (using alpha).
        """
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|
