import os
import glob
import numpy as np
import random
import torch 
from random import sample, choice
from h5image import H5Image
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image  # For loading images

class PatchGenerator(Dataset):
    def __init__(self, data_dir, maps, batch_size =32, patch_size = 256, norm_type='imagenet', valid_patch_rate = 0.75,
                    augment=True, vertical_flip_rate=0.4, horizontal_flip_rate=0.4, rotation_rate=0.4, hue_factor=0.2, num_workers=4):
        """
        Args:
            data_dir (str): Path to the directory containing HDF5 map files.
            train_split (float): Proportion of maps to use for the training set (between 0 and 1).
            **kwargs: Additional arguments to be passed to the PatchDataset class.
        """
        self.data_dir = data_dir
        self.maps = maps
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.valid_patch_rate = valid_patch_rate
        self.augment = augment
        self.batch_size = batch_size
        self.norm_type = norm_type
        self.valid_patch_rate = valid_patch_rate
        self.augment = augment
        self.vertical_flip_rate = vertical_flip_rate
        self.horizontal_flip_rate = horizontal_flip_rate
        self.rotation_rate = rotation_rate
        self.hue_factor = hue_factor  
        self.num_workers = num_workers

        if (norm_type=="imagenet"):
            self.data_transforms = transforms.Compose([
                                                        transforms.Resize((patch_size, patch_size)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225]),
                                                        ])
        else:
            self.resize_transform = transforms.Compose([
                                                        transforms.Resize((patch_size, patch_size)),
                                                        transforms.ToTensor(),
                                                        ])
        
        self.mapsh5i = self._load_maps()
        self.patches = self._generate_patches(self.maps)

        print("***Training Initialization ***") 
        print("self.data_dir :", self.data_dir)
        print("********************************")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        map_name, layer_name, row, column = self.patches[idx]
        data, label = self._get_data_and_label(map_name, layer_name, row, column, self.norm_type)
        return data, label
        
    def _load_maps(self):
        """Loads HDF5 map files from the data directory."""
        mapsh5i = {}
        for file in glob.glob(os.path.join(self.data_dir, "*.hdf5")):
            mapname = os.path.basename(file).replace(".hdf5", "")
            mapsh5i[mapname] = H5Image(file, "r")
        print("Setup complete. Number of maps loaded:", len(mapsh5i))
        return mapsh5i

    def _generate_patches(self, map_names):
        map_patches = []
        for map_name in map_names:
            if map_name in self.mapsh5i:
                patches = self.mapsh5i[map_name].get_patches(map_name) 
                cor = self.mapsh5i[map_name].get_map_corners(map_name) 
                valid_patches = [(map_name, layer, x, y) for layer in patches if layer.endswith('_poly') for x, y in patches[layer]]
                valid_layers = [key for key in patches.keys() if key.endswith('_poly')]
                random_patches = [(map_name, choice(valid_layers), 
                                sample(range(cor[0][0], cor[1][0]), 1)[0], 
                                sample(range(cor[0][1], cor[1][1]), 1)[0]) 
                                for _ in range(len(valid_patches))]
                n = len(valid_patches)
                percent_valid = self.valid_patch_rate
                map_patches += sample(valid_patches, int(n * percent_valid)) + sample(random_patches, int(n * (1 - percent_valid)))

        return map_patches

    def _get_data_and_label(self, map_name, layer_name, row, column, norm_type):

        # Get map patch
        map_patch = self.mapsh5i[map_name].get_patch(row, column, map_name)
        map_patch = self.data_transforms(map_patch)

        # Get legend patch
        legend_patch = self.mapsh5i[map_name].get_legend(map_name, layer_name)
        legend_patch_resize =self.data_transforms(legend_patch)

        # Get label
        label_patch = self.mapsh5i[map_name].get_patch(row, column, map_name, layer_name)

        # Augment data and label
        if self.augment:
            if random.random() < self.hue_factor:
                hue_delta = random.uniform(-self.hue_factor, self.hue_factor)
                map_hue = TF.adjust_hue(map_patch, hue_delta).numpy()
                legend_hue = TF.adjust_hue(legend_patch_resize, hue_delta).numpy()
                data_patch = np.concatenate([map_hue, legend_hue], axis=-1)

        # Normalize the map patch and legend
        if norm_type == 'imagenet':
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3)
            map_patch = ((torch.from_numpy(map_patch) / 255.0) - mean) / std
            legend_patch_resize = ((torch.from_numpy(legend_patch_resize) / 255.0) - mean) / std
            data_patch = torch.cat([map_patch, legend_patch_resize], dim=-1)
            
        elif norm_type == 'basic':
            data_patch = np.concatenate([map_patch, legend_patch_resize], axis=-1)
            data_patch = torch.from_numpy(data_patch) / 255.0
            
        else:
            data_patch = np.concatenate([map_patch, legend_patch_resize], axis=-1)
            data_patch = torch.from_numpy(data_patch) / 255.0

        # Augment data and label
        if self.augment:
            if random.random() < self.vertical_flip_rate:
                data_patch = torch.flip(data_patch, dims=[0])
                label_patch = torch.flip(label_patch, dims=[0])

            if random.random() < self.horizontal_flip_rate:
                data_patch = torch.flip(data_patch, dims=[1])
                label_patch = torch.flip(label_patch, dims=[1])

            if random.random() < self.rotation_rate:
                angle = random.randint(0, 3)  # PyTorch uses 90-degree increments
                data_patch = torch.rot90(data_patch, k=angle, dims=[0, 1])
                label_patch = torch.rot90(label_patch, k=angle, dims=[0, 1])

        return data_patch, label_patch
    
