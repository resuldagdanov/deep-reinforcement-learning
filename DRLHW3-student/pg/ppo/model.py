from typing import Optional, Tuple, List, Generator, Callable
from collections import namedtuple
import numpy as np
import torch

from pg.a2c.model import A2C
from pg.a2c.vecenv import ParallelEnv


class PPO(A2C):
    """ PPO agent.

    Args:
        network (torch.nn.Module): Policy and Value network (one network with two heads)
        args (namedtuple): Hyperparameters
        vecenv (ParallelEnv): Vectorized environment
        optimizer (torch.optim.Optimizer): Optimizer for both the network
    """

    Transition = namedtuple("Transition", "reward done state, action old_log_prob value")
    Rollout = namedtuple("Rollout", "list target_value")
    TrainData = namedtuple("TrainData", "old_log_prob advantage returns state action")

    def __init__(self,
                 network: torch.nn.Module,
                 args: namedtuple,
                 vecenv: ParallelEnv,
                 optimizer: torch.optim.Optimizer):
        super().__init__(
            network=network,
            args=args,
            vecenv=vecenv,
            optimizer=optimizer
        )

    def forward(self,
                state: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Return action, log probability of that action, value, and the entropy of the policy distribution.
        This function first generates a policy distribution pi(a|s) and calculates the remaning tensors using
        this distribution.

        Args:
            state (torch.Tensor): State tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                sampled action, log probability of the sample action, value of the given state, and
                entropy of the action distribution
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

    def forward_given_actions(self,
                              state: torch.Tensor,
                              action: torch.Tensor,
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass that does not return action samples. Instead, log probability and entropy is
        calculated for the given actions. We use this function to pass the same rollout data multiple times.

        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                log probability of the sample action, value of the given state, and entropy of 
                the action distribution for the given state and the action tensors
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError
        return log_prob, value, entropy

    @staticmethod
    def rollout_data_loader(rollout: Rollout,
                            advantages: List[torch.Tensor],
                            returns: List[torch.Tensor],
                            batch_size: int
                            ) -> Generator[TrainData, None, None]:
        """ Return a generator that yields mini batches that are randomly sampled
        from the rollout data.

        Args:
            rollout (Rollout): rollout sample
            advantages (List[torch.Tensor]): List of advantages
            returns (List[torch.Tensor]): List of returns
            batch_size (int): Mini batch size

        Yields:
            Generator[TrainData, None, None]: Yield mini batches of size <batch_size>
                from the rollout data
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

    def parameter_update(self,
                         rollout_data: Generator[TrainData, None, None],
                         clip_range: float,
                         ) -> Tuple[float, float, float]:
        """ Update the parameters by applying PPO update.

        Args:
            rollout_data (Generator[TrainData, None, None]): Flatten rollout generator to 
                compute the loss
            clip_range (float): PPO cliping range

        Returns:
            Tuple[float, float, float]: value loss, policy loss, and entropy loss for
                logging purposes only. Do not forget to detach loss tensors before
                returning.
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

    def collect_rollout(self,
                        states: np.ndarray,
                        ) -> Tuple[Rollout, np.ndarray]:
        """ Sample a rollout from the environment (vectorized environment) using online policy.

        Args:
            states (np.ndarray): Last state of the previous rollout

        Returns:
            Tuple[Rollout, np.ndarray]: Rollout object that includes transition
                data for n-step and the last state of the rollout
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

    def learn(self) -> None:
        """ Main loop of the training procedure. Initiates the training and log at every <write_period> """
        clip_schedule = self.linear_annealing(
            self.args.clip_range_max,
            self.args.clip_range_min,
            self.args.timesteps // (self.args.n_step * self.args.nenv))

        last_states = self.vecenv.reset()
        for timestep in range(0, self.args.timesteps, self.args.n_step * self.args.nenv):

            clip_range = next(clip_schedule)

            rollout, last_states = self.collect_rollout(last_states)
            advantages, returns = self.calculate_gae(
                rollout, self.args.gamma, self.args.gae_lambda)
            rollout_data_generator = self.rollout_data_loader(
                rollout, advantages, returns, self.args.batch_size)
            value_loss, policy_loss, entropy_loss = self.parameter_update(
                rollout_data_generator, clip_range)

            # Write to logger
            if timestep % self.args.write_period == (self.args.write_period - self.args.nenv * self.args.n_step):
                for writer in self.writers:
                    writer(dict(
                        timestep=timestep,
                        episodic_reward=np.mean(
                            self.vecenv.episodic_rewards[-self.args.log_window_length:]),
                        value_loss=value_loss,
                        policy_loss=policy_loss,
                        entropy_loss=entropy_loss,
                        clip_range=clip_range,
                    ))

    def evaluate(self,
                 envmaker: Callable,
                 n_episodes: int = 5,
                 device: str = "cpu"
                 ) -> float:
        """ Evaluate the agent loaded from the given path (if given) n_episodes many
        times and return the average undiscounted episodic reward.

        Args:
            envmaker (Callable): Environment returning function
            n_episodes (int, optional): Number of episodes of evaluation. Defaults to 5.
            device (str, optional): Device name. Defaults to "cpu".

        Returns:
            float: Average undiscounted episodic reward of <n_episodes> many evaluations
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

    @staticmethod
    def linear_annealing(init_value: float,
                         min_value: float,
                         decay_steps: float
                         ) -> Generator[float, None, None]:
        """ Decay and yield the value at every call linearly.

        Args:
            init_value (float): Initial value
            min_value (float): Minimum value
            decay_steps (float): Range of the decaying process in terms of
                iterations.

        Yields:
            Generator[float, None, None]: Yield annealed value
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError
