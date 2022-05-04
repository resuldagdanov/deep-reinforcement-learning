from typing import Dict, Any, Optional
import torch
import numpy as np
import math


class HeadLayer(torch.nn.Module):
    """
        Multi-function head layer. Structure of the layer changes depending on the "extensions" dictionary.
        If "noisy" is active, Linear layers become Noisy Linear.
        If "dueling" is active, the dueling architecture must be employed, and lastly,
        if "distributional" is active, output shape should change accordingly.

        Args:
            in_size (int): Input size of the head layer
            act_size (int): Action size
            extensions (Dict[str, Any]): A dictionary that keeps extension information
            hidden_size (Optional[int], optional): Size of the hidden layer in Dueling architecture. Defaults to None.

        Raises:
            ValueError: if hidden_size is not given while dueling is active
    """

    def __init__(self, in_size: int, act_size: int, extensions: Dict[str, Any], hidden_size: Optional[int] = None):
        super().__init__()

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
            Run last layer with the given features 

            Args:
                features (torch.Tensor): Input to the layer

            Returns:
                torch.Tensor: Q values or distributions
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

    def reset_noise(self) -> None:
        """
            Call reset_noise function of all child layers. Only used when "noisy" is active.
        """

        for module in self.children():
            module.reset_noise()


class NoisyLinear(torch.nn.Module):
    """
        Linear Layer with Noisy parameters. Noise level is set initially and kept fixed until "reset_noise" function is called.
        In training mode, noisy layer works stochastically while in evaluation mode it works as a standard Linear layer (using mean parameter values).

        Args:
            in_size (int): Input size
            out_size (int): Outout size
            init_std (float): Initial Standard Deviation
    """

    def __init__(self, in_size: int, out_size: int, init_std: float):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.init_std = init_std

        # average weigths
        self.weight_mu = torch.nn.Parameter(torch.Tensor(self.out_size, self.in_size))
        # standard deviation of weights
        self.weight_sigma = torch.nn.Parameter(torch.Tensor(self.out_size, self.in_size))
        self.register_buffer("weight_epsilon", torch.Tensor(self.out_size, self.in_size))

        # average biases
        self.bias_mu = torch.nn.Parameter(torch.Tensor(self.out_size))
        # standard deviation of biases
        self.bias_sigma = torch.nn.Parameter(torch.Tensor(self.out_size))
        self.register_buffer("bias_epsilon", torch.Tensor(self.out_size))

        self.reset_noise()

    def reset_noise(self) -> None:
        """ 
            Reset Noise of the parameters
        """
        
        # define range of noise
        mu_range = 1 / math.sqrt(self.in_size)

        # reset weights given the range
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.init_std / math.sqrt(self.in_size))
        
        # reset biases given the range
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.init_std / math.sqrt(self.out_size))

        # scaling factor for input noise
        in_epsilon = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=self.in_size))
        scale_in_epsilon = in_epsilon.sign().mul(in_epsilon.abs().sqrt())

        # scaling factor for output noise
        out_epsilon = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=self.out_size))
        scale_out_epsilon = out_epsilon.sign().mul(out_epsilon.abs().sqrt())

        # outer product
        self.weight_epsilon.copy_(scale_out_epsilon.ger(scale_in_epsilon))
        self.bias_epsilon.copy_(scale_out_epsilon)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """
            Forward function that works stochastically in training mode and deterministically in eval mode.

            Args:
                input (torch.Tensor): Layer input

            Returns:
                torch.Tensor: Layer output
        """
        
        return torch.nn.functional.linear(
            _input,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
