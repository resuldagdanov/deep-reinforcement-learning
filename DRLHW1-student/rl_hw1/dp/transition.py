""" Produce the necessary transition map for dynamic programming.
    Author: Your-name / Your-number
"""
from collections import defaultdict
import numpy as np


def make_transition_map(initial_board):
    """ Return the transtion map for passable(other than wall cells) states.
    In a state S an action A is chosen, there are four possibilities:
    - Intended action can be picked
    - 3 of the remaning action can be picked by the environment.
    Structure of the map:

    map[S][A] -> [
        (p_0, n_s, r_0, t_0), # Quad tuple of transition for the action 0
        (p_1, n_s, r_1, t_1), # Quad tuple of transition for the action 1
        (p_2, n_s, r_2, t_2), # Quad tuple of transition for the action 2
        (p_3, n_s, r_3, t_3), # Quad tuple of transition for the action 3
    ]

    p_x denotes the probability of transition by action "x"
    r_x denotes the reward obtained during the transition by "x"
    t_x denotes the termination condition at the new state(next state)
    n_s denotes the next state

    S denotes the space of all the non-wall states
    A denotes the action space which is range(4)
    So each value in map[S][A] is a length 4 list of quad tuples.


    Arguments:
        - initial_board: Board of the Mazeworld at initialization

    Return:
        transition map
    """
    #  ______   _____   _        _
    # |  ____| |_   _| | |      | |
    # | |__      | |   | |      | |
    # |  __|     | |   | |      | |
    # | |       _| |_  | |____  | |____
    # |_|      |_____| |______| |______|
