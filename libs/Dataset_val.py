import os
import glob
import h5py
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from h5image import H5Image
from random import choice, sample
from skimage.transform import resize
from .Normalization import BasicNormalizer, ImageNetNormalizer, MinMaxNormalizer

class PatchDataGenerator:
    def __init__(self, data_dir='/projects/bbym/shared/data/commonPatchData/', patch_size=256, overlap=15, norm_type='imagenet', valid_patch_rate=0.75,
                    augment=True, vertical_flip_rate=0.4, horizontal_flip_rate=0.4, rotation_rate=0.4, hue_factor=0.2, test=False ):

        self.patch_size = patch_size
        self.overlap = overlap
        self.data_dir = os.path.join(data_dir, str(self.patch_size), str(self.overlap))
        self.valid_patch_rate = valid_patch_rate
        self.augment = augment
        self.norm_type = norm_type
        self.valid_patch_rate = valid_patch_rate
        self.augment = augment
        self.vertical_flip_rate = vertical_flip_rate
        self.horizontal_flip_rate = horizontal_flip_rate
        self.rotation_rate = rotation_rate
        self.hue_factor = hue_factor
        self.test=test
        self.legend_trasform =transforms.Compose([transforms.ToPILImage(), transforms.Resize((self.patch_size, self.patch_size))])

        self.mapsh5i, self.map_names = self._load_maps(self.data_dir)
        self.patches = self._generate_patches(self.map_names, self.mapsh5i)

        print("*****Training Initialization ****")
        print(" data dir :", self.data_dir)
        print("*********************************")

    def _load_maps(self, h5_dir):
        mapsh5i = {}

        # List of filenames to exclude
        exclude_files = [
            "24_Black Crystal_2014_p43.hdf5",
            "26_Black Crystal_2018_6.hdf5",
            "46_Coosa_2015_11 74.hdf5",
            "DMEA2328_OldLeydenMine_CO.hdf5",
            "AZ_PrescottNF_basemap.hdf5",
            "CarlinVanadiumNI43-101Jun2020.hdf5",
            "Genesis_fig7_1.hdf5",
            "Genesis_fig9_1.hdf5",
            "Genesis_fig9_2.hdf5",
            "Genesis_fig9_3.hdf5",
            "Genesis_fig9_4.hdf5",
            "Genesis_fig9_5.hdf5",
            "Titac_2011_fig7_6.hdf5",
            "Trend_2005_fig4_3.hdf5",
            "Trend_2005_fig6_4.hdf5",
            "Trend_2005_fig7_2.hdf5",
            "Trend_2007_fig10_2.hdf5",
            "USGS_GF-7_2.hdf5"
        ]

        files_name = [f for f in glob.glob(os.path.join(h5_dir, "*.hdf5")) if os.path.basename(f) not in exclude_files]

        if self.test: #for testing purpose
            files_name = files_name[:3]
            print(files_name)

        for file in files_name:
            mapname = os.path.basename(file).replace(".hdf5", "")
            mapsh5i[mapname] = H5Image(file, "r")

        print("Setup complete. Number of maps loaded:", len(mapsh5i))
        return mapsh5i, list(mapsh5i.keys())

    def _generate_patches(self, map_names, h5image_obj):
        map_patches = []

        for map_name in map_names:
            if map_name in h5image_obj:
                patches = h5image_obj[map_name].get_patches(map_name)
                cor = h5image_obj[map_name].get_map_corners(map_name)

                # Filter patches based on label presence:
                valid_patches = [
                    (map_name, layer, row, column)
                    for layer in patches if layer.endswith('_poly')
                    for row, column in patches[layer]
                    if np.sum(h5image_obj[map_name].get_patch(row, column, map_name, layer)) >= 100
                ]

                # Sample patches with a mix of valid and random locations:
                valid_layers = [key for key in patches.keys() if key.endswith('_poly')]
                random_patches = [
                    (map_name, choice(valid_layers),
                    random.randint(cor[0][0], cor[1][0]),
                    random.randint(cor[0][1], cor[1][1]))
                    for _ in range(len(valid_patches))
                ]

                n = len(valid_patches)
                valid_sample_size = int(n * self.valid_patch_rate)
                random_sample_size = int(n * (1 - self.valid_patch_rate))

                map_patches.extend(sample(valid_patches, valid_sample_size))
                map_patches.extend(sample(random_patches, random_sample_size))

        return map_patches

    def _augment_data(self, map_patch, legend_patch, label_patch):

        # Vertical Flip
        if random.random() < self.vertical_flip_rate:
            map_patch = TF.vflip(map_patch)
            legend_patch = TF.vflip(legend_patch)
            label_patch = TF.vflip(label_patch)

        # Horizontal Flip
        if random.random() < self.horizontal_flip_rate:
            # Apply the transformation
            map_patch = TF.hflip(map_patch)
            legend_patch = TF.hflip(legend_patch)
            label_patch = TF.hflip(label_patch)

        # Rotation (by 90 degrees)
        if random.random() < self.rotation_rate:
            rotation_angle = random.choice([90, 180, 270])
            map_patch = TF.rotate(map_patch, rotation_angle)
            legend_patch = TF.rotate(legend_patch, rotation_angle)
            label_patch = TF.rotate(label_patch, rotation_angle)

        # Color Jitter
        if random.random() < self.hue_factor:
            random_brightness = random.uniform(0.8, 1.2)
            map_patch = TF.adjust_brightness(map_patch, brightness_factor=random_brightness)
            # map_patch = TF.adjust_hue(map_patch, hue_factor=self.hue_factor)
            # legend_patch = TF.adjust_hue(legend_patch, hue_factor=self.hue_factor)

        return map_patch, legend_patch, label_patch

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        # print(self.patches[index])
        map_name, layer_name, row, column = self.patches[index]
        # Get map patch (using h5py syntax)
        map_patch = torch.from_numpy(self.mapsh5i[map_name].get_patch(row, column, map_name))
        map_patch = map_patch.permute(2, 0, 1)
        # print("map_patch", map_patch.shape)

        # Get legend patch
        legend_patch = self.mapsh5i[map_name].get_legend(map_name, layer_name)
        # print("legend_patch:", legend_patch.shape)
        legend_patch = resize(legend_patch, (self.patch_size, self.patch_size), preserve_range= True, anti_aliasing=True)
        # print("legend_patch resized:", legend_patch.shape)
        legend_patch = torch.from_numpy(legend_patch)
        legend_patch_resize = legend_patch.permute(2, 0, 1)
        # legend_patch_resize = self.legend_trasform(legend_patch)
        # print("legend_patch_resize", legend_patch_resize.shape)

        # Get label
        label_patch = torch.from_numpy(self.mapsh5i[map_name].get_patch(row, column, map_name, layer_name))
        # Add new axis to label patch
        label_patch = label_patch.unsqueeze(0)
        # print("label_patch", label_patch.shape)

        # Apply augmentation
        if self.augment:
            # map_patch, legend_patch, label_patch = map_patch, legend_patch_resize, label_patch
            map_patch, legend_patch, label_patch = self._augment_data(map_patch, legend_patch_resize, label_patch)

        # Normalize the map patch and legend
        if self.norm_type == 'imagenet':
            normalizer = ImageNetNormalizer()
            map_patch_norm = normalizer.normalize(map_patch)
            legend_patch_norm = normalizer.normalize(legend_patch)
            data_patch = np.concatenate([map_patch_norm, legend_patch_norm], axis=0)

        elif self.norm_type == 'basic':
            normalizer = MinMaxNormalizer(0, 255, -1, 1)
            data_patch = np.concatenate([map_patch, legend_patch], axis=0)
            data_patch = normalizer.normalize(data_patch)

        else:
            normalizer = BasicNormalizer()
            data_patch = np.concatenate([map_patch, legend_patch], axis=0)
            data_patch = normalizer.normalize(data_patch)

        return data_patch, label_patch