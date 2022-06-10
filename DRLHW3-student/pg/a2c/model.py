from typing import Optional, Tuple, List, Generator, Callable
from collections import namedtuple
import os
import numpy as np
import torch
import gym

from pg.common import PrintWriter, CSVwriter
from pg.a2c.vecenv import ParallelEnv


class A2C(torch.nn.Module):
    """
        Advantage Actor Critic agent (with Synchronous training). Learning is performed using vectorized environments.

        Args:
            network (torch.nn.Module): Policy and Value network (one network with two heads)
            args (namedtuple): Hyperparameters
            vecenv (ParallelEnv): Vectorized parallel environments
            optimizer (torch.optim.Optimizer): Optimizer for the network
    """

    Transition = namedtuple("Transition", "reward done state action log_prob value entropy")
    Rollout = namedtuple("Rollout", "list target_value")
    TrainData = namedtuple("TrainData", "log_prob advantage returns value entropy")

    def __init__(self,
                 network: torch.nn.Module,
                 args: namedtuple,
                 vecenv: ParallelEnv,
                 optimizer: torch.optim.Optimizer):
        super().__init__()
        
        self.network = network
        self.args = args
        self.vecenv = vecenv
        self.optim = optimizer
        
        if args is not None:
            self.writers = [PrintWriter(flush=True), CSVwriter(self.args.log_dir)]

    def forward(self, state: torch.Tensor, *args: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
            Return action, log probability of that action, value, the entropy of the policy distribution,
            and the reccurrent state representation if available. This function first generates a policy distribution
            pi(a|s) and calculates the remaning tensors using this distribution.

            Args:
                state (torch.Tensor): State tensor

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: 
                    sampled action, log probability of the sample action, value of the given state, entropy of
                    the action distribution and the reccurrent hidden state if available
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
    def to_torch(array: np.ndarray, device: str) -> torch.Tensor:
        """
            Helper method that transforms numpy array to torch tensor

            Args:
                array (np.ndarray): Numpy array to transform
                device (str): Device name

            Returns:
                torch.Tensor: Torch array of the given numpy array
        """

        return torch.from_numpy(array).to(device).float()

    @staticmethod
    def calculate_gae(rollout: Rollout, gamma: float, gae_lambda: float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
            Generalized Advantage Estimation (GAE). Compute the advantage
            and return list by applying GAE procedure. Advantage list will be used in
            policy loss while the return list will be used for the value loss.

            Args:
                rollout (Rollout): Sampled rollout
                gamma (float): Discount factor
                gae_lambda (float): Lambda coefficient in the Eligibility Trace 

            Returns:
                Tuple[List[torch.Tensor], List[torch.Tensor]]: List of advantages, List of returns
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
    def rollout_data_loader(rollout: Rollout, advantages: List[torch.Tensor], returns: List[torch.Tensor]) -> TrainData:
        """
            Return a <TrainData> object that contains flatten tensors. This method converts rollout
            data to flatten tensors to be used in training.

            Args:
                rollout (Rollout): Sampled rollout. Contains list of transition info.
                advantages (List[torch.Tensor]): List of advantages. Contains list of per transition advantages.
                returns (List[torch.Tensor]): List of returns. Contains list of per transition returns.

            Returns:
                TrainData: flatten tensors of (log_prob, advantage, returns, value, and entropy). Return shape
                    of the tensors: (B * T, 1) where B denotes batch size and T denotes rollout length.
        """

        rewards, dones, states, actions, log_probs, values, entropies = [
            torch.cat(tensor, dim=0) for tensor in zip(*rollout.list)
        ]
        advantages = torch.cat(advantages, dim=0)
        returns = torch.cat(returns, dim=0)
        return A2C.TrainData(
            log_probs,
            advantages,
            returns,
            values,
            entropies,
        )

    def parameter_update(self, rollout_data: TrainData) -> Tuple[float, float, float]:
        """
            Update the parameters by computing the gradients with respect to the
            loss function that includes policy, value and entropy losses.

            Args:
                rollout_data (TrainData): Flatten rollout information to compute the loss

            Returns:
                Tuple[float, float, float]: value loss, policy loss, and entropy loss for
                    logging purposes only. Do not forget to detach loss tensors before
                    returning. Otherwise it may cause memory leakages.
        """

        # Dont forget to detach returns and advantages
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError

    def collect_rollout(self, states: np.ndarray, gru_hidden: torch.Tensor) -> Tuple[Rollout, np.ndarray, torch.Tensor]:
        """
            Sample a rollout from the environment (vectorized parallel environments) using the online policy.

            Args:
                states (np.ndarray): Last state of the previous rollout
                gru_hidden (torch.Tensor): Last gru hidden output of the previous rollout. To avoid
                    unwanted gradient propagation, do not forget to detach it at the begining

            Returns:
                Tuple[Rollout, np.ndarray, torch.Tensor]: Rollout object that includes n-step 
                transition data, the last state of the rollout, and the last hidden state of GRU
        """

        # Do not forget to reset gru_hidden if the corresponding index (environment) is reset
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
        """
            Main loop of the training procedure. Initiates the training and log at every <write_period>
        """

        gru_hx = torch.zeros((self.args.nenv, self.args.hidden_size), device=self.args.device)

        last_states = self.vecenv.reset()
        for timestep in range(0, self.args.timesteps, self.args.n_step * self.args.nenv):

            rollout, last_states, gru_hx = self.collect_rollout(last_states, gru_hx)
            advantages, returns = self.calculate_gae(
                rollout, self.args.gamma, self.args.gae_lambda)
            rollout_data_generator = self.rollout_data_loader(
                rollout, advantages, returns)
            value_loss, policy_loss, entropy_loss = self.parameter_update(rollout_data_generator)

            # Write to logger
            if timestep % self.args.write_period == (self.args.write_period - self.args.nenv * self.args.n_step):
                for writer in self.writers:
                    writer(dict(
                        timestep=timestep,
                        episodic_reward=np.mean(
                            self.vecenv.episodic_rewards[-self.args.log_window_length:]),
                        value_loss=value_loss.item(),
                        policy_loss=policy_loss.item(),
                        entropy_loss=entropy_loss.item(),
                    ))

    def save(self, path: str) -> None:
        """
            Save the model parameters
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "args": {"hidden_size": self.args.hidden_size}}, path)

    def evaluate(self, envmaker: Callable[[], gym.Env], n_episodes: int = 5, device: str = "cpu") -> float:
        """
            Evaluate the agent loaded from the given path (if given) n_episodes many
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
