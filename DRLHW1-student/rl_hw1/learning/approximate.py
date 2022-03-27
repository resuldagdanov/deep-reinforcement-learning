"""
    Function apporximation methods in TD
    
    Author: Resul Dagdanov / 511211135
"""

import numpy as np
import time
import random
from collections import namedtuple


class ApproximateAgent():
    """
        Base class for the approximate methods. This class provides policies and training loop.
        Initiate a weight matrix with shape (#observation x #action).
    """

    def __init__(self, nobs, nact):
        self.nact = nact
        self.nobs = nobs

        self.weights = np.random.uniform(-0.1, 0.1, size=(nobs, nact))

    def q_values(self, state):
        """
            Return the q values of the given state for each action.
        """

        return np.dot(state, self.weights)

    def greedy_policy(self, state, *args):
        """
            Return the best possible action according to the value function
        """

        return np.argmax(self.q_values(state))

    def e_greedy_policy(self, state, epsilon):
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

    def train(self, env, policy, args):
        """
            Training loop for the approximate agents. Initiate an episodic reward list and a loss list.
            At each episode decrease the epsilon value exponentially using args.eps_decay_rate
            within the boundries of the args.init_eps and args.final_eps. At each transition update
            the agent (weights of the function). For every "args._evaluate_period"'th step call evaluation function
            and store the returned episodic reward to the reward list.

            Arguments:
                - env: gym environment
                - policy: Behaviour policy to be used in training (not in evaluation)
                - args: namedtuple of hyperparameters

            Return:
                - Episodic reward list of evaluations (not the training rewards)
                - Loss list of the training (one loss for per update)
        """

        # fix seeds
        random.seed(args.seed)
        np.random.seed(args.seed)

        # initiate an episodic reward list
        list_reward = []
        list_loss = []

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

                # update the value function using given transition by returning mean squered td error loss
                loss = self.update(transition=transition, alpha=args.alpha, gamma=args.gamma)
                list_loss.append(loss)

                obs = next_obs
                action = next_action
            
            # call evaluation function and store episodic reward
            if eps % args.evaluate_period == 0:
                list_reward.append(episodic_reward)
            
        return list_reward, list_loss
    
    def update(self, *arg, **kwargs):
        raise NotImplementedError

    def evaluate(self, env):
        raise NotImplementedError


class ApproximateQAgent(ApproximateAgent):
    """
        Approximate Q learning agent where the learning is done via minimizing the mean squared
        value error with semi-gradient descent. This is an off-policy algorithm.
    """

    def __init__(self, nobs, nact):
        super().__init__(nobs, nact)

    def update(self, transition, alpha, gamma):
        """
            Update the parameters that parameterized the value function according to (semi-gradient) q learning.

            Arguments:
                - transition: 4 tuple of state, action, reward and next_state
                - alpha: Learning rate of the update function
                - gamma: Discount rate

            Return:
                Mean squared temporal difference error
        """

        # open the given transition tuple
        state, action, reward, next_state, next_action = transition

        # calculate gradient of weights using numpy function
        gradients = np.array(np.gradient(self.weights)[0])

        # compute mean squared temporal-difference error with gradients
        mean_td_err = gradients * (reward + (gamma * self.q_values(next_state)[next_action]) - self.q_values(state)[action])

        # apply temporal-difference update method to optimize weight
        self.weights = self.weights + (alpha * mean_td_err)

        return mean_td_err

    def evaluate(self, env, render=False):
        """
            Single episode evaluation of the greedy agent.
        
            Arguments:
                - env: gym environemnt
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
                env.render(mode='close')
        
        env.close()

        return episode_reward
