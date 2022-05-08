from typing import Generator
from collections import namedtuple, deque
from functools import reduce
import argparse
import gym
import torch

from .model import RainBow
from dqn.common import linear_annealing, exponential_annealing, PrintWriter
from dqn.dqn.train import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    """
        Training class that organize evaluation, update, and transition gathering.
            Arguments:
                - args: Parser arguments
                - agent: RL agent object
                - opt: Optimizer that optimizes agent's parameters
                - env: Gym environment
    """

    def __init__(self, args: argparse.Namespace, agent: RainBow, opt: torch.optim.Optimizer, env: gym.Env):
        """
            Training class that organize evaluation, update, and transition gathering.

            Args:
                args (argparse.Namespace): CL arguments
                agent (RainBow): RainBow Agent
                opt (torch.optim.Optimizer): Optimizer for agents parameters
                env (gym.Env): Environment
        """

        super().__init__(args, agent, opt, env)

        # beta = 1 - prioritized_beta
        # self.prioritized_beta = linear_annealing(
        #     init_value=1 - args.beta_init,
        #     min_value=0,
        #     decay_range=args.n_iterations
        # )

    def update(self, iteration: int) -> None:
        """
            One step updating function. Update the agent in training mode.
            - clip gradient if "clip_grad" is given in args.
            - keep track of td loss. Append td loss to "self.td_loss" list
            - Update target network.

            If the prioritized buffer is active:
                - Use the weighted average of the loss where the weights are returned by the prioritized buffer
                - Update priorities of the sampled transitions

            If noisy-net is active:
                - reset noise for valuenet and targetnet
            Check for the training index "iteration" to start the update.

            Args:
                iteration (int): Training iteration
        """
        
        self.agent.train()

        # before updating make sure that the number of stored transitions are greater than the batch size
        if iteration >= self.args.batch_size:

            # checking whether updating initialization starting iteration is reached
            if iteration >= self.args.start_update:

                # reset an optimizer
                self.opt.zero_grad()

                if self.args.no_prioritized:
                    # sample a transitions from the replay buffer in size of a mini-batch
                    transitions = self.agent.buffer.sample(self.args.batch_size)

                    # compute mean temporal-difference loss of this batch (with averaging)
                    avg_loss = self.agent.loss(transitions, self.args.gamma).mean()
                
                else:
                    # sample a transitions from the prioritized replay buffer along with weigths and indices
                    transitions, selected_indices, weights = self.agent.buffer.sample(batch_size=self.args.batch_size, beta=self.args.beta_init)

                    # compute temporal-difference loss of this batch (without averaging)
                    loss = self.agent.loss(transitions, self.args.gamma)

                    # filter out negative td-losses
                    loss[loss < 0.0] = 0.0

                    weights = torch.tensor(weights).to(self.args.device)
                    loss = loss * weights

                    td_values = loss + 1e-6

                    # averaging td-loss with weights
                    avg_loss = torch.mean(loss)

                    self.agent.buffer.update_priority(indices=selected_indices, td_values=td_values.detach().cpu().numpy())

                # keeping track of the mean td-loss
                self.td_loss.append(avg_loss.item())

                # backpropagate with computed loss through value network
                avg_loss.backward()

                # clip each parameter of the value network between [-1 1] if required
                if self.args.clip_grad:
                    for param in self.agent.valuenet.parameters():
                        param.grad.data.clamp_(-1, 1)
                
                # update optimizer
                self.opt.step()

                # update target network with given frequency period
                if (iteration % self.args.target_update_period) == 0:
                    self.agent.update_target()

                # reset noise for value and target network
                if not self.args.no_noisy:
                    self.agent.valuenet.head_layer.reset_noise()
                    self.agent.targetnet.head_layer.reset_noise()

    def __iter__(self) -> Generator[RainBow.Transition, None, None]:
        """
            n-step transition generator. Yield a transition with n-step look ahead. Use the greedy policy if noisy network extension is activate.

            Yields:
                Generator[RainBow.Transition, None, None]: Transition of (s_t, a_t, \sum_{j=t}^{t+n}(\gamma^{j-t} r_j), done, s_{t+n})
        """
        
        # initialize total reward for this episode
        episode_reward = 0.0

        # horizon for each episode
        n_steps = self.args.n_steps

        # initialize episode step counter
        step = 0

        # loop over steps in episode
        done = False
        while not done:

            # initialize cumulative reward computed with discount factor
            cumulative_reward = 0.0

            # store current state for yielding in transition
            yielding_state = self.state

            # loop in an episode
            for step in range(n_steps):

                if self.args.no_noisy:
                    # current epsilon-greedy (stochastic) policy is evaluated
                    action = self.agent.e_greedy_policy(state=torch.Tensor(self.state).to(self.args.device), epsilon=self.epsilon_value)
                else:
                    # current greedy policy is evaluated
                    action = self.agent.greedy_policy(state=torch.Tensor(self.state).to(self.args.device))
                
                # yield first action
                if step == 0:
                    yielding_action = action
                
                # step in the environment with this epsilon-greedy action
                next_state, reward, done, _ = self.env.step(action)

                # accumulate reward
                episode_reward += reward

                # cumulative reward for the next states
                cumulative_reward += reward * (self.args.gamma ** (step + 1))

                # termination state
                yielding_done = done

                # termination
                if done:
                    # get updated epsilon
                    self.epsilon_value = next(self.epsilon)

                    # reset the environment
                    self.state = self.env.reset()
                    self.train_rewards.append(episode_reward)

                    episode_reward = 0.0
                    done = False

                    break
            
                # continue with next transition steps if not terminated
                else:
                    self.state = next_state

            yield self.agent.Transition(yielding_state, yielding_action, cumulative_reward, next_state, yielding_done)
