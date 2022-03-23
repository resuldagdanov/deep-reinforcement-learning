""" Tabular TD methods
    Author: Your-name / Your-number
"""
from collections import defaultdict
from collections import namedtuple
import random
import math
import numpy as np
import time
from collections import deque

from .monte_carlo import TabularAgent


class TabularTDAgent(TabularAgent):
    """ Base class for Tabular TD agents for shared training loop.
    """

    def train(self, env, policy, args):
        """ Training loop for tabular td agents.
        Initiate an episodic reward list. At each episode decrease the epsilon
        value exponentially using args.eps_decay_rate within the boundries of
        args.init_eps and args.final_eps. For every "args._evaluate_period"'th
        step call evaluation function and store the returned episodic reward
        to the list.

        Arguments:
            - env: Warehouse environment
            - policy: Behaviour policy to be used in training(not in
            evaluation)
            - args: namedtuple of hyperparameters

        Return:
            - Episodic reward list of evaluations (not the training rewards)

        **Note**: This function will be used in both Sarsa and Q learning.
        **Note** that: You can also implement you own answer to question 10.
        """
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|


class QAgent(TabularTDAgent):
    """ Tabular Q leanring agent. Update rule is based on Q learning.
    """

    def __init__(self, nact):
        super().__init__(nact)

    def update(self, transition, alpha, gamma):
        """ Update values of a state-action pair based on the given transition
        and parameters.

        Arguments:
            - transition: 5 tuple of state, action, reward, next_state and
            next_action. "next_action" will not be used in q learning update.
            It is there to be compatible with SARSA update in "train" method.
            - alpha: Exponential decay rate of updates
            - gamma: Discount ratio

        Return:
            temporal diffrence error
        """
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|


class SarsaAgent(TabularTDAgent):
    """ Tabular Sarsa agent. Update rule is based on
    SARSA(State Action Reward next_State, next_Action).
    """

    def __init__(self, nact):
        super().__init__(nact)

    def update(self, trans, alpha, gamma):
        """ Update values of a state-action pair based on the given transition
        and parameters.

        Arguments:
            - transition: 5 tuple of state, action, reward, next_state and
            next_action.
            - alpha: Exponential decay rate of updates
            - gamma: Discount ratio

        Return:
            temporal diffrence error
        """
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|
