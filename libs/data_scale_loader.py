import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from h5image import H5Image
from random import choice, sample
from skimage.transform import resize
from .Normalization import BasicNormalizer, ImageNetNormalizer, MinMaxNormalizer
import torchvision.transforms.functional as TF

class ScaleDataGenerator(Dataset):
    def __init__(self, data_dir, batch_size=32, patch_size=256, overlap=15, norm_type='basic', valid_patch_rate=0.75,
                 augment=True, vertical_flip_rate=0.4, horizontal_flip_rate=0.4, rotation_rate=0.4, hue_factor=0.2, test_flag=False):
        
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.valid_patch_rate = valid_patch_rate
        self.augment = augment
        self.norm_type = norm_type
        self.vertical_flip_rate = vertical_flip_rate
        self.horizontal_flip_rate = horizontal_flip_rate
        self.rotation_rate = rotation_rate
        self.hue_factor = hue_factor
        self.test_flag = test_flag

        self.patch_sizes = [128, 256, 512, 1024]  # Supported patch sizes

        self.mapsh5i, self.map_names = self._load_maps(self.data_dir)

        self.patches = self._generate_patches(self.map_names, self.mapsh5i)

    def _load_maps(self, h5_dir):
        mapsh5i = {}
        exclude_files = [
            "24_Black Crystal_2014_p43.hdf5", "26_Black Crystal_2018_6.hdf5",
            "46_Coosa_2015_11 74.hdf5", "DMEA2328_OldLeydenMine_CO.hdf5",
            "AZ_PrescottNF_basemap.hdf5", "CarlinVanadiumNI43-101Jun2020.hdf5",
            "Genesis_fig7_1.hdf5", "Genesis_fig9_1.hdf5", "Genesis_fig9_2.hdf5",
            "Genesis_fig9_3.hdf5", "Genesis_fig9_4.hdf5", "Genesis_fig9_5.hdf5",
            "Titac_2011_fig7_6.hdf5", "Trend_2005_fig4_3.hdf5", "Trend_2005_fig6_4.hdf5",
            "Trend_2005_fig7_2.hdf5", "Trend_2007_fig10_2.hdf5", "USGS_GF-7_2.hdf5",
            "CA_SanJone"
        ]

        for size in self.patch_sizes:
            sub_dir = os.path.join(h5_dir, str(size), str(self.overlap))
            size_mapsh5i = {}
            files_name = [f for f in glob.glob(os.path.join(sub_dir, "*.hdf5")) if os.path.basename(f) not in exclude_files]
            for file in files_name:
                path = os.path.join(sub_dir, file)
                mapname = os.path.basename(path).replace(".hdf5", "")
                size_mapsh5i[mapname] = H5Image(path, "r")
            mapsh5i[size] = size_mapsh5i  
        map_names = list(mapsh5i[self.patch_size].keys())  # Default to input patch size for map_names

        if(self.test_flag):
            map_names = map_names[:2]

        return mapsh5i, map_names

    def _generate_patches_for_map(self, map_name, mapsh5i):
        map_patches = []
        expected_num_patches=0
        for size in self.patch_sizes:
            if map_name in mapsh5i[size]:
                patches = mapsh5i[size][map_name].get_patches(map_name)
                cor = mapsh5i[size][map_name].get_map_corners(map_name)

                valid_patches = []
                for layer in patches:
                    if layer.endswith('_poly'):
                        for row, column in patches[layer]:
                            label_patch = mapsh5i[size][map_name].get_patch(row, column, map_name, layer)
                            if np.sum(label_patch) >= 100: 
                                valid_patches.append((map_name, layer, row, column, size)) 

                valid_layers = [key for key in patches.keys() if key.endswith('_poly')]
                random_patches = [(map_name, choice(valid_layers), 
                                    sample(range(cor[0][0], cor[1][0]), 1)[0],  
                                    sample(range(cor[0][1], cor[1][1]), 1)[0],
                                    size)  
                                    for _ in range(len(valid_patches))]
                
                n = len(valid_patches)
                if(size==self.patch_size):
                    expected_num_patches = n
                map_patches += sample(valid_patches, int(n * self.valid_patch_rate)) + sample(random_patches, int(n * (1 - self.valid_patch_rate)))

        return map_patches, expected_num_patches

    def _generate_patches(self, map_names, mapsh5i):
        all_patches = []
        for map_name in map_names:
            map_patches, target_num_patches = self._generate_patches_for_map(map_name, mapsh5i)
            print("map_name:", map_name, "target_num_patches:", target_num_patches, "map_patches:", len(map_patches))

            selected_patches = []
            while len(selected_patches) < target_num_patches:
                patch = random.choice(map_patches)
                if patch not in selected_patches:
                    selected_patches.append(patch)
                    map_patches.remove(patch)

            all_patches.extend(selected_patches)
        return all_patches

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
    
    def _get_data_and_label(self, map_name, layer_name, row, column, h5image_obj):
        
        # map patch
        map_patch = h5image_obj[map_name].get_patch(row, column, map_name)
        map_patch = resize(map_patch, (self.patch_size, self.patch_size), preserve_range= True, anti_aliasing=True)
        map_patch = torch.from_numpy(map_patch).float()
        map_patch = map_patch.permute(2, 0, 1)
        
        # Get legend patch
        legend_patch = h5image_obj[map_name].get_legend(map_name, layer_name)
        legend_patch = resize(legend_patch, (self.patch_size, self.patch_size), preserve_range= True, anti_aliasing=True)
        legend_patch = torch.from_numpy(legend_patch).float()
        legend_patch = legend_patch.permute(2, 0, 1)

        # Get label
        label_patch = h5image_obj[map_name].get_patch(row, column, map_name, layer_name)
        label_patch = resize(label_patch, (self.patch_size, self.patch_size), preserve_range= True, anti_aliasing=True)
        # Binarize the label patch
        threshold = 0.5
        label_patch = (label_patch >= threshold).astype(np.uint8)
        label_patch = torch.from_numpy(label_patch)
        # Add new axis to label patch
        label_patch = label_patch.unsqueeze(0)

        # Apply augmentation
        if self.augment:
            map_patch, legend_patch, label_patch = self._augment_data(map_patch, legend_patch, label_patch)

        # Normalize the map patch and legend
        if self.norm_type == 'imagenet':
            normalizer = ImageNetNormalizer()
            map_patch_norm = normalizer.normalize(map_patch)
            legend_patch_norm = normalizer.normalize(legend_patch)
            data_patch = torch.cat([map_patch_norm, legend_patch_norm], dim=0)

        elif self.norm_type == 'basic':
            normalizer = MinMaxNormalizer(0, 255, -1, 1)
            data_patch = torch.cat([map_patch, legend_patch], dim=0)
            data_patch = normalizer.normalize(data_patch)

        else:
            normalizer = BasicNormalizer()
            data_patch = torch.cat([map_patch, legend_patch], dim=0)
            data_patch = normalizer.normalize(data_patch)

        return data_patch.float(), label_patch.float()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        map_name, layer_name, row, column, size = self.patches[idx]
        data, label = self._get_data_and_label(map_name, layer_name, row, column, self.mapsh5i[size])
        return data, label

    # @staticmethod
    # def create_train_dataset(batch_size=32, patch_size=256, overlap=15, norm_type='imagenet', valid_patch_rate=0.75,
    #                          augment=True, vertical_flip_rate=0.4, horizontal_flip_rate=0.4, rotation_rate=0.4, hue_factor=0.2, test_flag=False):
    #     train_dir = '/projects/bcxi/nathanj/commonPatchData/'
    #     return ScaleDataGenerator._load_maps_static(train_dir, patch_size, overlap, test_flag)

    # @staticmethod
    # def create_validation_dataset(batch_size=32, patch_size=256, overlap=15, norm_type='imagenet', valid_patch_rate=0.75,
    #                               augment=True, vertical_flip_rate=0.4, horizontal_flip_rate=0.4, rotation_rate=0.4, hue_factor=0.2, test_flag=False):
    #     validation_dir = '/projects/bcxi/nathanj/commonPatchData/validation/'
    #     mapsh5i, map_names = ScaleDataGenerator._load_maps_static(validation_dir, patch_size, overlap, test_flag)
    #     return DataLoader(ScaleDataGenerator(map_names, mapsh5i, batch_size, patch_size, overlap, norm_type, valid_patch_rate,
    #                                          augment, vertical_flip_rate, horizontal_flip_rate, rotation_rate, hue_factor, test_flag), 
    #                       batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # @staticmethod
    # def _load_maps_static(h5_dir, patch_size, overlap, test_flag=False):
    #     mapsh5i = {}
    #     exclude_files = [
    #         "24_Black Crystal_2014_p43.hdf5", "26_Black Crystal_2018_6.hdf5",
    #         "46_Coosa_2015_11 74.hdf5", "DMEA2328_OldLeydenMine_CO.hdf5",
    #         "AZ_PrescottNF_basemap.hdf5", "CarlinVanadiumNI43-101Jun2020.hdf5",
    #         "Genesis_fig7_1.hdf5", "Genesis_fig9_1.hdf5", "Genesis_fig9_2.hdf5",
    #         "Genesis_fig9_3.hdf5", "Genesis_fig9_4.hdf5", "Genesis_fig9_5.hdf5",
    #         "Titac_2011_fig7_6.hdf5", "Trend_2005_fig4_3.hdf5", "Trend_2005_fig6_4.hdf5",
    #         "Trend_2005_fig7_2.hdf5", "Trend_2007_fig10_2.hdf5", "USGS_GF-7_2.hdf5",
    #         "CA_SanJone"
    #     ]

    #     patch_sizes = [128, 256, 512, 1024]  # Supported patch sizes

    #     for size in patch_sizes:
    #         sub_dir = os.path.join(h5_dir, str(size), str(overlap))
    #         size_mapsh5i = {}
    #         files_name = [f for f in glob.glob(os.path.join(sub_dir, "*.hdf5")) if os.path.basename(f) not in exclude_files]
    #         for file in files_name:
    #             path = os.path.join(sub_dir, file)
    #             mapname = os.path.basename(path).replace(".hdf5", "")
    #             size_mapsh5i[mapname] = H5Image(path, "r")
    #         mapsh5i[size] = size_mapsh5i  

    #     map_names = list(mapsh5i[patch_size].keys())  # Default to input patch size for map_names
        
    #     if(test_flag):
    #         map_names = map_names[:2]

    #     return mapsh5i, map_names
