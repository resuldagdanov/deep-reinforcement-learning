from typing import Generator
import torch
import numpy as np
from copy import deepcopy
import argparse
import gym
import os
from tempfile import TemporaryDirectory
import warnings

from dqn.replaybuffer.uniform import UniformBuffer

from .model import DQN
from dqn.common import linear_annealing, exponential_annealing, PrintWriter, CSVwriter


class Trainer:
    """
        Training class that organize evaluation, update, and transition gathering.

        Args:
            args (argparse.Namespace): CL arguments
            agent (DQN): DQN Agent
            opt (torch.optim.Optimizer): Optimizer for agents parameters
            env (gym.Env): Environment
    """

    def __init__(self, args: argparse.Namespace, agent: DQN, opt: torch.optim.Optimizer, env: gym.Env):
        self.env = env
        self.args = args
        self.agent = agent
        self.opt = opt

        self.train_rewards = []
        self.eval_rewards = []
        self.td_loss = []
        self.log_dir = args.log_dir

        if self.log_dir is None:
            self.log_dir = TemporaryDirectory().name
            warnings.warn("Temporary Logging directory: {}".format(self.log_dir))
            
        self._writers = [PrintWriter(flush=True), CSVwriter(self.log_dir)]

        self.checkpoint_reward = -np.inf
        self.agent.to(args.device)

        if args.epsilon_decay is not None:
            self.epsilon = exponential_annealing(
                args.epsilon_init,
                args.epsilon_min,
                args.epsilon_decay
            )
        else:
            self.epsilon = linear_annealing(
                args.epsilon_init,
                args.epsilon_min,
                args.n_iterations if args.epsilon_range is None else args.epsilon_range
            )
        
        self.state = self.env.reset()
        self.epsilon_value = args.epsilon_init

    def __call__(self) -> None:
        """
            Start training
        """
        
        for iteration, trans in enumerate(self):
            self.evaluation(iteration)

            self.agent.push_transition(trans)

            self.update(iteration)
            self.writer(iteration)

    def evaluation(self, iteration: int) -> None:
        """
            Evaluate the agent if the index "iteration" equals to the evaluation period. 
            If "save_model" is given the current best model is saved based on the evaluation score.
            Evaluation score appended into the "eval_rewards" list to keep track of evaluation scores.

            Args:
                iteration (int): Training iteration

            Raises:
                FileNotFoundError:  If "save_model" is given in arguments and directory given by "model_dir" does not exist
        """

        if iteration % self.args.eval_period == 0:

            self.eval_rewards.append(
                self.agent.evaluate(self.args.eval_episode, self.env, self.args.device, self.args.render))

            if self.eval_rewards[-1] > self.checkpoint_reward and self.args.save_model:
                self.checkpoint_reward = self.eval_rewards[-1]
                
                model_id = "{}_{:6d}_{:6.3f}.b".format(
                    self.agent.__class__.__name__,
                    iteration,
                    self.eval_rewards[-1]).replace(" ", "0")
                
                if not os.path.exists(self.args.model_dir):
                    raise FileNotFoundError(
                        "No directory as {}".format(self.args.model_dir))
                
                torch.save(dict(model=self.agent.state_dict(), optim=self.opt.state_dict(),),
                    os.path.join(self.args.model_dir, model_id))

    def update(self, iteration: int) -> None:
        """
            One step updating function. Update the agent in training mode, clip gradient if "clip_grad" is given in args,
            and keep track of td loss. Check for the training index "iteration" to start the update.

            Append td loss to "self.td_loss" list

            Args:
                iteration (int): Training iteration
        """

        self.agent.train()

        # before updating make sure that the number of stored transitions are greater than the batch size
        if iteration >= self.args.batch_size:
            
            # checking whether updating initialization starting iteration is reached
            if iteration >= self.args.start_update:

                # sample a transitions from the replay buffer in size of a mini-batch
                transitions = self.agent.buffer.sample(self.args.batch_size)

                # reset an optimizer
                self.opt.zero_grad()

                # compute temporal-difference loss of this batch
                loss = self.agent.loss(transitions, self.args.gamma)

                # keeping track of the td-loss
                self.td_loss.append(loss.item())

                # backpropagate with computed loss through value network
                loss.backward()

                # clip each parameter of the value network between [-1 1] if required
                if self.args.clip_grad:
                    for param in self.agent.valuenet.parameters():
                        param.grad.data.clamp_(-1, 1)
                
                # update optimizer
                self.opt.step()

                # update target network with given frequency period
                if (iteration % self.args.target_update_period) == 0:
                    self.agent.update_target()

    def writer(self, iteration: int) -> None:
        """
            Simple writer function that feed PrintWriter with statistics 

            Args:
                iteration (int): Training iteration
        """
        
        if iteration % self.args.write_period == 0:
            for _writer in self._writers:
                _writer(
                    {
                        "Iteration": iteration,
                        "Train reward": np.mean(self.train_rewards[-20:]),
                        "Eval reward": self.eval_rewards[-1],
                        "TD loss": np.mean(self.td_loss[-100:]),
                        "Episode": len(self.train_rewards),
                        "Epsilon": self.epsilon_value
                    })

    def __iter__(self) -> Generator[UniformBuffer.Transition, None, None]:
        """
            Experience collector function that yields a transition at every iteration for "args.n_iterations"
            iterations by collecting experience from the environment. If the environment terminates,
            append the episodic reward and reset the environment. 

            Append episodic reward to "self.train_rewards" at every termination
        """

        # initialize total reward for this episode
        episode_reward = 0.0

        # loop over steps in episode
        done = False
        while not done:

            # store current state for yielding in transition
            yielding_state = self.state

            # current epsilon-greedy (stochastic) policy is evaluated
            action = self.agent.e_greedy_policy(state=torch.Tensor(self.state).to(self.args.device), epsilon=self.epsilon_value)
            
            # step in the environment with this epsilon-greedy action
            next_state, reward, done, _ = self.env.step(action)

            # accumulate reward
            episode_reward += reward

            # termination
            if done:
                yielding_done = True

                # get updated epsilon
                self.epsilon_value = next(self.epsilon)

                # reset the environment
                self.state = self.env.reset()
                self.train_rewards.append(episode_reward)

                episode_reward = 0.0
                done = False
            
            # continue with next transition steps if not terminated
            else:
                yielding_done = False
                self.state = next_state

            yield self.agent.Transition(yielding_state, action, reward, next_state, yielding_done)
