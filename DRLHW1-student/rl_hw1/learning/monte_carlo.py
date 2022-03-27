"""
    Tabular MC algorithms
    
    Author: Resul Dagdanov / 511211135
"""

from collections import defaultdict
import random
import math
from collections import deque
import numpy as np
import time


class TabularAgent:
    """
        Based Tabular Agent class that inludes policies and evaluation function
    """

    def __init__(self, nact):
        self.qvalues = defaultdict(lambda: [0.0] * nact)
        self.nact = nact

    def greedy_policy(self, state, *args, **kwargs):
        """
            Policy that returns the best action according to q values.
        """

        # returns each action to determine each action's q-value
        q_values = self.qvalues[state]

        # greedy action is the action resulting in maximum q-value
        max_q_action = q_values.index(max(q_values))
        return max_q_action

    def e_greedy_policy(self, state, epsilon, *args, **kwargs):
        """
            Policy that returns the best action according to q values with (epsilon/#action) + (1 - epsilon)
            probability and any other action with probability episolon/#action.
        """

        # generate random number between [0 - 1]
        rand_number = random.random()

        # select random action (exploration)
        if rand_number <= epsilon:
            action = random.randrange(start=0, stop=self.nact, step=1)
        
        # select greedy action (exploitation)
        else:
            action = self.greedy_policy(state=state)

        return action

    def evaluate(self, env, render=False):
        """
            Single episode evaluation of the greedy agent.
            
            Arguments:
                - env: Warehouse or Mazeworld environemnt
                - render: If true render the environment(default False)
            
            Return:
                Episodic reward
        """

        done = False
        episode_reward = 0.0
        
        # initialize environment reset
        obs = env.reset()

        # loop one episode until termination state is reached
        while done is False:

            # step a greedy action in the environment
            action = self.greedy_policy(state=obs)
            obs, reward, done, info = env.step(action)

            # cumulative episodic reward
            episode_reward += reward

            # whether to visualize environment or not
            if render:
                env.render()
        
        return episode_reward


class MonteCarloAgent(TabularAgent):
    """
        Tabular Monte Carlo Agent that updates q values based on MC method.
    """

    def __init__(self, nact):
        super().__init__(nact)

    def one_episode_train(self, env, policy, gamma, alpha):
        """
            Single episode training function.
            
            Arguments:
                - env: Mazeworld environment
                - policy: Behaviour policy for the training loop
                - gamma: Discount factor
                - alpha: Exponential decay rate of updates

            Returns:
                episodic reward

            **Note** that in the book (Sutton & Barto), they directly assign the return to q value.
            You can either implement that algorithm (given in chapter 5) or use exponential decaying update (using alpha).
        """
        
        done = False
        episode_reward = 0.0
        episode_transitions = []

        # initialize environment reset
        obs = env.reset()

        # loop one episode until termination state to collect transition trajectories
        while done is False:

            # step in the environment with given policy
            action = policy(obs)
            next_obs, reward, done, info = env.step(action)

            # cumulative episodic reward
            episode_reward += reward

            # state, action, reward, done -> tuple transition
            transition = (obs, action, reward, done)
            episode_transitions.append(transition)

            obs = next_obs

        # episode total return
        G = 0
        returns = []
        
        episode_transitions.reverse()

        # loop for each step of generated episode
        for transition in episode_transitions:

            state = transition[0]
            action = transition[1]
            reward = transition[2]
            done = transition[3]

            # cummulatively update total return
            G = reward + (gamma * G)
            returns.append(G)

            # take an average of total returns
            avg_returns = sum(returns) / len(returns)

            # update q value function
            self.qvalues[state][action] += avg_returns

        return episode_reward