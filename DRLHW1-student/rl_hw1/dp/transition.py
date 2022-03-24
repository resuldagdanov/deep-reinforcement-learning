"""
    Produce the necessary transition map for dynamic programming.
    
    Author: Resul Dagdanov / 511211135
"""

from collections import defaultdict
import numpy as np


def make_transition_map(initial_board):
    """
        Return the transtion map for passable (other than wall cells) states.
        In a state S an action A is chosen, there are four possibilities:
        - Intended action can be picked
        - 3 of the remaning action can be picked by the environment.
        
        Structure of the map:
        map[S][A] -> [
            (p_0, n_s, r_0, t_0), # Quad tuple of transition for the action 0 -> up
            (p_1, n_s, r_1, t_1), # Quad tuple of transition for the action 1 -> down
            (p_2, n_s, r_2, t_2), # Quad tuple of transition for the action 2 -> left
            (p_3, n_s, r_3, t_3), # Quad tuple of transition for the action 3 -> right
        ]

        p_x denotes the probability of transition by action "x"
        r_x denotes the reward obtained during the transition by "x"
        t_x denotes the termination condition at the new state(next state)
        n_s denotes the next state

        S denotes the space of all the non-wall states
        A denotes the action space which is range(4)
        So each value in map[S][A] is a length 4 list of quad tuples.

        state 32: passable
        state 35: non-passable (wall)
        state 80: terminal

        Arguments:
            - initial_board: Board of the Mazeworld at initialization

        Return:
            transition map
    """

    # width and height of the environment
    world_shape = initial_board.shape
    height, width = world_shape

    # total number of grids and action in the environment
    n_states = height * width
    n_actions = 4

    # get coordinates for each state that is passable (35 means non-passible)
    passable_states = np.argwhere(np.asarray(initial_board) != 35)

    # get coordinates of the state grid that are non-passable wall states
    walls = np.argwhere(np.asarray(initial_board) == 35)

    # get state coordinate for episode termination
    terminal_states = np.argwhere(np.asarray(initial_board) == 80)

    transition_map = {}

    # make an empty dictionary of (state, action) pairs
    for state in passable_states.tolist():
        transition_map[tuple(state)] = {}

        for action in range(n_actions):
            transition_map[tuple(state)][action] = []

    # function to check whether selected grid is a wall grid or not
    def check_wall_state(x, y):
        if [y, x] in walls.tolist():
            return True
        else:
            return False

    # function to check whether termination state is reached in next state
    def check_termination(x, y):
        if [y, x] in terminal_states.tolist():
            return 1.0, True
        else:
            return 0.0, False

    # loop through each state transition
    for current_h in range(height):
        for current_w in range(width):

            # tuple (x, y) coordinate of current state
            current_state = (current_h, current_w)

            # ignore being on position in the wall state
            if check_wall_state(*current_state):
                continue

            # loop through all actions on each passable state
            for action in range(n_actions):

                transition_map[current_state][action] = []

                # loop through all possible stochastic actions for one state
                for prob_action in range(n_actions):

                    # up
                    if prob_action == 0:
                        next_state_h = current_h - 1
                        next_state_w = current_w
                    
                    # down
                    elif prob_action == 1:
                        next_state_h = current_h + 1
                        next_state_w = current_w
                    
                    # left
                    elif prob_action == 2:
                        next_state_h = current_h
                        next_state_w = current_w - 1
                    
                    # right
                    elif prob_action == 3:
                        next_state_h = current_h
                        next_state_w = current_w + 1
                    
                    else:
                        print("[error]: action is out of bounds !")
                    
                    # tuple (x, y) coordinate of next state
                    next_state = (next_state_h, next_state_w)

                    # actions that lead to wall state are ignored
                    # and next state will be the current state
                    if check_wall_state(*next_state):
                        next_state = current_state

                    # sparse reward is only obtained when termination state is reached
                    reward, terminate = check_termination(*next_state)

                    # only one %70 probabilistic action is applicable in each state
                    if prob_action == action:
                        prabability = 0.7
                    else:
                        prabability = 0.1

                    # state transition tuple
                    transition = (prabability, next_state, reward, terminate)

                    # store each state transition tuple
                    transition_map[current_state][action].append(transition)

    return transition_map