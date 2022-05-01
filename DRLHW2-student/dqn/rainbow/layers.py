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

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/

    def reset_noise(self) -> None:
        """ 
            Reset Noise of the parameters
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
            Forward function that works stochastically in training mode and deterministically in eval mode.

            Args:
                input (torch.Tensor): Layer input

            Returns:
                torch.Tensor: Layer output
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
