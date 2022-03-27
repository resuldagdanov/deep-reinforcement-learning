"""
    Tabular TD methods
    
    Author: Resul Dagdanov / 511211135
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
    """
        Base class for Tabular TD agents for shared training loop.
    """

    def train(self, env, policy, args):
        """
            Training loop for tabular td agents. Initiate an episodic reward list.
            At each episode decrease the epsilon value exponentially using args.eps_decay_rate within
            the boundries of args.init_eps and args.final_eps. For every "args._evaluate_period"'th step
            call evaluation function and store the returned episodic reward to the list.

            Arguments:
                - env: Warehouse environment
                - policy: Behaviour policy to be used in training (not in evaluation)
                - args: namedtuple of hyperparameters

            Return:
                - Episodic reward list of evaluations (not the training rewards)

            **Note**: This function will be used in both Sarsa and Q learning.
            **Note** that: You can also implement you own answer to question 10.
        """

        # fix seeds
        random.seed(args.seed)
        np.random.seed(args.seed)

        # initiate an episodic reward list
        list_episode_reward = []

        # initial epsilon value which will be changed in each episode
        epsilon = args.init_eps

        # loop through each episode
        for eps in range(args.episodes):
            done = False

            # initialize total reward obtained in one episode
            episodic_reward = 0.0
            
            # calculate epsilon with exponential decay
            epsilon = max(min(args.init_eps, epsilon * args.eps_decay_rate), args.final_eps)

            # first environment reset
            obs = env.reset()

            # get epsilon-greedy policy action for initial state observation
            action = self.e_greedy_policy(state=obs, epsilon=epsilon)

            # loop until terminate state is reached
            while done is False:

                # step in the enviroment to get one step transition
                next_obs, reward, done, info = env.step(action)
                episodic_reward += reward

                # next action in the transition is calculated with current policy (epsilon-greedy)
                next_action = policy(state=next_obs, epsilon=epsilon)

                transition = (obs, action, reward, next_obs, next_action)

                # update the value function using given transition
                self.update(transition=transition, alpha=args.alpha, gamma=args.gamma)

                obs = next_obs
                action = next_action

            # call evaluation function and store episodic reward
            if eps % args.evaluate_period == 0:
                list_episode_reward.append(episodic_reward)

        return list_episode_reward


class QAgent(TabularTDAgent):
    """
        Tabular Q learning agent. Update rule is based on Q learning.
    """

    def __init__(self, nact):
        super().__init__(nact)

    def update(self, transition, alpha, gamma):
        """
            Update values of a state-action pair based on the given transition and parameters.

            Arguments:
                - transition: 5 tuple of state, action, reward, next_state and
                  next_action. "next_action" will not be used in q learning update.
                  It is there to be compatible with SARSA update in "train" method.
                - alpha: Exponential decay rate of updates
                - gamma: Discount ratio

            Return:
                temporal diffrence error
        """

        # open the given transition tuple
        state, action, reward, next_state, next_action = transition

        # next state's action is taken as a greedy action
        greedy_next_action = self.greedy_policy(next_state)

        # temporal difference error is calculated
        td_err = reward + (gamma * self.qvalues[next_state][greedy_next_action]) - self.qvalues[state][action]

        # apply temporal difference update method
        self.qvalues[state][action] = self.qvalues[state][action] + (alpha * td_err)

        return td_err


class SarsaAgent(TabularTDAgent):
    """
    Tabular Sarsa agent. Update rule is based on SARSA (State Action Reward next_State, next_Action).
    """

    def __init__(self, nact):
        super().__init__(nact)

    def update(self, transition, alpha, gamma):
        """
            Update values of a state-action pair based on the given transition and parameters.

            Arguments:
                - transition: 5 tuple of state, action, reward, next_state and next_action.
                - alpha: Exponential decay rate of updates
                - gamma: Discount ratio

            Return:
                temporal diffrence error
        """
        
        # one step transition tuple
        state, action, reward, next_state, next_action = transition

        # temporal difference error is calculated with only using transition tuple variables
        td_err = reward + (gamma * self.qvalues[next_state][next_action]) - self.qvalues[state][action]

        # apply temporal difference update method
        self.qvalues[state][action] = self.qvalues[state][action] + (alpha * td_err)

        return td_err
