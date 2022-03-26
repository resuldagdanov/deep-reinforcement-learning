"""
    This module includes DP algorithms.
    
    Author: Resul Dagdanov / 511211135
"""


class DPAgent():
    """
        Base Dynamic Programming class. DP methods requires the transition map in order to optimize policies.
        This class provides the policy and one step policy evaluation as well as policy improvement.
        It serves as a base class for Policy Iteration and Value Iteration algorithms.
    """

    def __init__(self, nact, transitions_map, init_value=0.0):
        self.nact = nact
        
        # environment model [state][action] : (probability, next state, reward, termination) tuples
        self.transitions_map = transitions_map

        # initialize value function values for each agent state
        self.values = {s: init_value for s in self.transitions_map.keys()}
        
        # initialize uniform policy distribution where agent has the same probability of doing an action in all states
        self.policy_dist = {s: [1.0 / nact] * nact for s in self.transitions_map.keys()}

    def policy(self, state):
        """
            Policy pi that returns the action with the highest Q value. You can use "policy_dist" that
            keeps probability values for each action at any state. You may have a stochastic policy(random.choice)
            or a determistic one(argmax). Your choice. But for this homework we will be using deterministic policy.
            Therefore, at each state only one action will be possible with probability 1.(Initially its stochastic)
            
            Example policy_dist:
                policy_dist[S] = [1, 0, 0, 0]

            Arguments:
                - state: State / observation of the environment

            Returns:
                action for the given state
        """

        # policy distribution for the given state
        distibution = self.policy_dist[state]

        # get maximum valued action from the distribution
        # acting greedy in finding maximum value action
        action = distibution.index(max(distibution))
        
        return action

    def one_step_policy_eval(self, gamma=0.95):
        """
            One step policy evaluation. You can follow the pseudocode given in the "Reinforcement Learing
            Book (Sutton & Barta) chapter 4.1" Remember to return delta: Maximum change among all the states

            Arguments:
                - gamma: Discount factor

            Return:
                delta
        """

        # initialize maximum amount of change among all possible states
        delta = 0.0

        # number of action at each state
        n_actions = self.nact

        # store each value update inside a buffer,
        # after all values in all states are calculated for one iteration, update original value function
        buffer_value = {}

        # loop through all passible states
        for state in self.values.keys():

            new_value = 0.0

            # loop through all possible actions
            for action in range(n_actions):
                transitions = self.transitions_map[state][action]

                # sum of all probability values of the Bellman equation for doing an action
                sum_bellman = 0.0

                # each possible transition for applying an action
                for trans in transitions:
                    probability = trans[0]
                    next_state = trans[1]
                    reward = trans[2]
                    terminate = trans[3]

                    # compute Bellman equation
                    sum_bellman += probability * (reward + gamma * self.values[next_state] * (1 - int(terminate)))

                # apply iteration policy evaluation with respect to action probability of the policy
                new_value += self.policy_dist[state][action] * sum_bellman
            
            # update delta (difference between new updated state and the old previous state value function)
            delta = max(delta, abs(self.values[state] - new_value))

            # update current value function
            buffer_value[state] = new_value

        # only update value function after one policy evaluation iteration is done in all states
        for state in self.values.keys():
            self.values[state] = buffer_value[state]
        
        return delta

    def policy_improvement(self, gamma=0.95):
        """
            Policy impovement updates the policy according to the most recent values.
            You can follow the ideas given in the "Reinforcement Learing (Sutton & Barta) chapter 4.2.
            Basically, look one step ahead to choose the best action. Remember to return a boolean value to state
            if the policy is stable. Also note that, improved policy must be deterministic.
            So that, at each state only one action is probable (like onehot vector).
            
            Example policy:
                policy_dist[S] = [0, 1, 0, 0]

            Arguments:
                - gamma: Discount factor

            Return:
                a boolean value stating if stability is reached
        """

        # stability of the policy
        is_stable = False

        # number of action at each state
        n_actions = self.nact

        # loop through all passible states
        for state in self.values.keys():

            # previous taken action for the same state
            prev_action = self.policy(state)

            # value of making each action
            action_values = []

            # loop through all possible actions
            for action in range(n_actions):
                transitions = self.transitions_map[state][action]

                # sum of all probability values of the Bellman equation for doing an action
                state_value = 0.0

                # each possible transition for applying an action
                for trans in transitions:
                    probability = trans[0]
                    next_state = trans[1]
                    reward = trans[2]
                    terminate = trans[3]

                    # compute Bellman equation
                    state_value += probability * (reward + gamma * self.values[next_state] * (1 - int(terminate)))

                # store value of making this action
                action_values.append(state_value)

            # counting of all values to normalize and find the probability of each action
            count_values = action_values.count(max(action_values))
   
            # compute action probabilities for each state
            for action in range(n_actions):

                # deterministic: make maximum value action 1 and others 0
                self.policy_dist[state][action] = 1 / count_values if action_values[action] == max(action_values) else 0
            
            # check the whether the policy is improved after acting greedy on updated value function
            if prev_action == self.policy(state):
                is_stable = True
        
        return is_stable


class PolicyIteration(DPAgent):
    """
        Policy Iteration algorithm that first evaluates the values until they converge within epsilon range,
        then updates the policy and repeats the process until the policy no longer changes.
    """

    def __init__(self, transitions_map, nact=4):
        super().__init__(nact, transitions_map)

    def optimize(self, gamma, epsilon=0.05, n_iteration=1):
        """
            This is the main function where you implement PI. Simply use "one_step_policy_eval" and
            "policy_improvement" methods. Itearte as "n_iteration" times in a loop.
            
            Arguments:
                - gamma: Discount factor
                - epsilon: convergence region of value evaluation
                - n_iteration: Number of iterations

            This should not take more than 10 lines.
        """

        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|


class ValueIteration(DPAgent):
    """
        Value Iteration algorithm iteratively evaluates the values and updates the policy until
        the values converges within epsilon range.
    """

    def __init__(self, transitions_map, nact=4):
        super().__init__(nact, transitions_map)

    def optimize(self, gamma, n_iteration=1):
        """
            This is the main function where you implement VI. Simply use "one_step_policy_eval" and
            "policy_improvement" methods. Itearte as "n_iteration" times in a loop.
            
            Arguments:
                - gamma: Discount factor
                - n_iteration: Number of iterations
            This should not take more than 5 lines.
        """
        
        #  ______   _____   _        _
        # |  ____| |_   _| | |      | |
        # | |__      | |   | |      | |
        # |  __|     | |   | |      | |
        # | |       _| |_  | |____  | |____
        # |_|      |_____| |______| |______|
