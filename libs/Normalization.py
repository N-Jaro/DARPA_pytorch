import numpy as np

class MinMaxNormalizer:
    def __init__(self, original_min=0, original_max=255, target_min=-1, target_max=1):
        self.original_min = original_min
        self.original_max = original_max
        self.target_min = target_min
        self.target_max = target_max

    def normalize(self, value):
        return ((value - self.original_min) / (self.original_max - self.original_min)) * (self.target_max - self.target_min) + self.target_min

class BasicNormalizer:
    def __init__(self, value):
        self.original_val = value

    def normalize(self, value):
        return self.original_val / 255.0

class ImageNetNormalizer:
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def normalize(self, array):
        """
        Normalize an array for ImageNet models. The input array is assumed to be in the range [0, 255].

        Parameters:
        - array: NumPy array to normalize, with the last dimension being 3 (for RGB channels).

        Returns:
        - The normalized NumPy array.
        """
        array = array.astype(np.float32) / 255.0
        for i in range(3):  # Assuming the last dimension of array is for channels
            array[..., i] = (array[..., i] - self.means[i]) / self.stds[i]
        return array

