import numpy as np
import torch

class MinMaxNormalizer:
    """
    Performs Min-Max normalization on PyTorch Tensors.
    """
    def __init__(self, min_val, max_val, min_target=0, max_target=1):
        self.min_val = min_val
        self.max_val = max_val
        self.min_target = min_target
        self.max_target = max_target

    def normalize(self, data):
        """
        Normalizes a PyTorch Tensor using Min-Max normalization.

        Args:
            data (torch.Tensor): The input Tensor to normalize.

        Returns:
            torch.Tensor: The normalized Tensor.
        """
        normalized_data = (data - self.min_val) / (self.max_val - self.min_val)
        return normalized_data * (self.max_target - self.min_target) + self.min_target

class BasicNormalizer:
    def __init__(self):
        self.divisor = torch.tensor(255.0, dtype=torch.float32)

    def normalize(self, tensor):
        """
        Normalize a tensor by dividing by 255.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return tensor / self.divisor

class ImageNetNormalizer:
    def __init__(self):
        self.means = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.stds = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def normalize(self, tensor):
        """
        Normalize a PyTorch Tensor for ImageNet models. Assumes input is in range [0, 255].

        Parameters:
        - tensor (torch.Tensor): Tensor to normalize, expected shape (C, W, H).

        Returns:
        - The normalized PyTorch Tensor.
        """
        tensor = tensor / 255.0  # Convert to float and scale

        # Ensure that means and stds are reshaped for (C, 1, 1) to broadcast properly
        means = self.means.view(-1, 1, 1)
        stds = self.stds.view(-1, 1, 1)

        # Perform normalization
        tensor = (tensor - means) / stds

        return tensor

