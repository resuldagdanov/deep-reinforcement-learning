""" Run multiple environments in parallel.

    Wrap your environment with ParallelEnv to run them in parallel. A wrapped
    parallel environment provides step and reset functions. After an initial
    reset call, there is no need for additional reset calls. If one of the
    environment terminates after the step call, it automatically resets the
    environment and sends the initial observation as next state. Thereby, users
    of this wrapper must be AWARE of the fact that if termination occurs
    next state becomes the initial observation of the new episode.
"""
from typing import Callable, Tuple, Dict, Any
from collections import namedtuple
import cloudpickle
import numpy as np
from torch.multiprocessing import Process, Pipe
import gym


class CloudpickleWrapper:
    """ Function wrapper

    Args:
        function (Any): function to wrap
    """

    def __init__(self, function: Callable):
        self.function = function

    def __getstate__(self) -> Callable:
        return cloudpickle.dumps(self.function)

    def __setstate__(self, function: Callable) -> None:
        self.function = cloudpickle.loads(function)


class ParallelEnv():
    """ Synchronized multiple environments wrapper.

        Workers communicate through pipes where each worker runs a single
        environment. Initiation is started by calling <start> method. After
        the initiating the workers, step function can be called indefinitely.
        In case of termination, each worker restarts it's own environment and
        returns the first state of the restarted environment instead of the
        last state of the terminated one.

        Args:
            n_envs (int): Number of parallel environments
            env_maker_fn (Callable): Environment returning function

        Example:
            >>> p_env = ParallelEnv(n, lambda: gym.make(env_name))
            >>> states = p_env.reset()
            >>>     actions = policy(states)
            >>>     for i in range(TIMESTEPS):
            >>>         states, rewards, dones = p_env.step(actions)
    """

    EnvProcess = namedtuple("EnvProcess", "process, remote")

    def __init__(self, n_envs: int, env_maker_fn: Callable, seed=None):
        self.seed = seed or np.random.randint(2**10, 2**30)
        env = env_maker_fn()
        self.action_space = env.action_space
        self.env_maker_fn = env_maker_fn
        self.n_envs = n_envs
        self.started = False

    @staticmethod
    def add_seed_wrapper(env_maker_fn, seed):
        env = env_maker_fn()
        env.seed(seed)
        return env

    def reset(self) -> np.ndarray:
        """ 
        Initiate the worker processes and return all the initial states.

        Raises:
            RuntimeError: If called twice without <close>

        Returns:
            np.ndarray: The first observations stacked as numpy array
        """

        if self.started is True:
            raise RuntimeError("cannot restart without closing")

        self.env_rewards = np.zeros((self.n_envs,))
        self.episodic_rewards = []

        env_processes = []
        for rank, (p_r, w_r) in enumerate((Pipe() for _ in range(self.n_envs))):
            process = Process(target=self.worker,
                              args=(w_r,
                                    CloudpickleWrapper(lambda: self.add_seed_wrapper(self.env_maker_fn, self.seed + rank))),
                              daemon=True)
            env_processes.append(self.EnvProcess(process, p_r))
            process.start()
            p_r.send("start")
        self.env_processes = env_processes

        state = np.stack(remote.recv() for _, remote in self.env_processes)
        self.started = True
        return state

    def step(self,
             actions: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """ Steps all the workers(environments) and return stacked
        observations, rewards, and termination arrays. When a termination
        happens in one of the workers, it returns the first observation of the
        restarted environment instead of returning the next-state of the
        terminated episode.

        Args:
            actions (np.ndarray): Action array for all the parallel environments.
                Shape: [B, a] where a denotes number of actions (1 if Discrete) and
                B denotes the batch size or the number of environments.

        Raises:
            RuntimeError: If called before start
            ValueError: If argument <actions> is not a 2D array
            ValueError: If the batch dimension of the array <actions> is not equal
                to the number of parallel environments

        Returns:
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]: Stacked arrays of 
                next_state, rewards, terminations and infos 
        """
        if self.started is False:
            raise RuntimeError("call <start> function first!")
        if len(actions.shape) != 2:
            raise ValueError("<actions> must be 2 dimensional!")
        if actions.shape[0] != self.n_envs:
            raise ValueError("not enough actions!")
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions = actions.squeeze(-1)
        for act, (_, remote) in zip(actions, self.env_processes):
            remote.send(act)

        state, reward, done = [np.stack(batch) for batch in zip(*(
            remote.recv() for _, remote in self.env_processes))]

        for index, (_done, _reward), in enumerate(zip(done, reward)):
            self.env_rewards[index] += _reward
            if _done:
                self.episodic_rewards.append(self.env_rewards[index])
                self.env_rewards[index] = 0

        return (state,
                reward.reshape(-1, 1).astype(np.float32),
                done.reshape(-1, 1).astype(np.float32))

    def close(self) -> None:
        """ Terminate and join all the workers. """
        for process, remote in self.env_processes:
            remote.send("end")
            process.terminate()
            process.join()
        self.started = False

    @staticmethod
    def worker(remote: Pipe,
               env_maker_fn: CloudpickleWrapper
               ) -> None:
        """ Start when the initial start signal is received from <reset>
        call. Following the start signal, the first observation array is
        sent through the pipe. Then, the worker waits for the action from the
        pipe. If the action is "end" string, then break the loop and terminate.
        Otherwise, the worker steps the environment and sends (state, reward,
        done) array triplet.

        Args:
            remote (Pipe):  Child pipe (for the worker)
            env_maker_fn (Callable): Function that returns the env object
        """
        env = env_maker_fn.function()
        state = env.reset()
        # Wait for the start command
        remote.recv()
        remote.send(state)
        while True:
            action = remote.recv()
            if isinstance(action, str) and action == "end":
                break
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
            remote.send((state, reward, done))
