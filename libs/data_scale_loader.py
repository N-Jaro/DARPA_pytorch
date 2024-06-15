import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from h5image import H5Image
from random import choice, sample
from skimage.transform import resize
from Normalization import MinMaxNormalizer, BasicNormalizer, ImageNetNormalizer
import torchvision.transforms.functional as TF

class ScaleDataGenerator(Dataset):
    def __init__(self, map_names, mapsh5i, batch_size=32, patch_size=256, overlap=15, norm_type='imagenet', valid_patch_rate=0.75,
                 augment=True, vertical_flip_rate=0.4, horizontal_flip_rate=0.4, rotation_rate=0.4, hue_factor=0.2, test_flag=False):

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

        self.mapsh5i = mapsh5i
        self.map_names = map_names

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

        print("Setup complete. Number of maps loaded:", len(mapsh5i))
        map_names = list(mapsh5i[self.patch_size].keys())  # Default to input patch size for map_names
        return mapsh5i, map_names

    def _generate_patches_for_map(self, map_name, mapsh5i):
        map_patches = []
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
                map_patches += sample(valid_patches, int(n * self.valid_patch_rate)) + sample(random_patches, int(n * (1 - self.valid_patch_rate)))

        return map_patches

    def _generate_patches(self, map_names, mapsh5i):
        all_patches = []
        for map_name in map_names:
            map_patches = self._generate_patches_for_map(map_name, mapsh5i)
            target_num_patches = len(mapsh5i[self.patch_size][map_name].get_patches(map_name))

            selected_patches = []
            while len(selected_patches) < target_num_patches:
                patch = random.choice(map_patches)
                if patch not in selected_patches:
                    selected_patches.append(patch)
                    map_patches.remove(patch)

            all_patches.extend(selected_patches)
        return all_patches

    def _augment_data(self, data_patch, label_patch):
        if random.random() < self.vertical_flip_rate:
            data_patch = np.flipud(data_patch)
            label_patch = np.flipud(label_patch)

        if random.random() < self.horizontal_flip_rate:
            data_patch = np.fliplr(data_patch)
            label_patch = np.fliplr(label_patch)

        if random.random() < self.rotation_rate:
            angle = random.choice([0, 90, 180, 270])
            data_patch = np.rot90(data_patch, k=angle // 90)
            label_patch = np.rot90(label_patch, k=angle // 90)

        if random.random() < self.hue_factor:
            hue_delta = random.uniform(-self.hue_factor, self.hue_factor)
            data_patch = TF.adjust_hue(torch.from_numpy(data_patch), hue_delta).numpy()

        return data_patch, label_patch
    
    def _get_data_and_label(self, map_name, layer_name, row, column, h5image_obj):
        map_patch = h5image_obj.get_patch(row, column, map_name)
        map_patch = TF.resize(torch.from_numpy(map_patch), [self.patch_size, self.patch_size]).numpy()

        legend_patch = h5image_obj.get_legend(map_name, layer_name)
        legend_patch = TF.resize(torch.from_numpy(legend_patch), [self.patch_size, self.patch_size]).numpy()

        label_patch = h5image_obj.get_patch(row, column, map_name, layer_name)
        label_patch = resize(label_patch, (self.patch_size, self.patch_size), order=0, preserve_range=True, anti_aliasing=False) 

        if self.norm_type == 'imagenet':
            normalizer = ImageNetNormalizer()
            map_patch = normalizer.normalize(map_patch)
            legend_patch = normalizer.normalize(legend_patch)

        elif self.norm_type == 'basic': 
            normalizer = MinMaxNormalizer(0, 255, -1, 1)
            map_patch = normalizer.normalize(map_patch)
            legend_patch = normalizer.normalize(legend_patch)

        else:  
            normalizer = BasicNormalizer()
            map_patch = normalizer.normalize(map_patch)
            legend_patch = normalizer.normalize(legend_patch)

        data_patch = np.concatenate([map_patch, legend_patch], axis=-1)

        if self.augment:
            data_patch, label_patch = self._augment_data(data_patch, label_patch)

        return data_patch, label_patch

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        map_name, layer_name, row, column, size = self.patches[idx]
        data, label = self._get_data_and_label(map_name, layer_name, row, column, self.mapsh5i[size])
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        return data, label

    @staticmethod
    def create_train_dataset(batch_size=32, patch_size=256, overlap=15, norm_type='imagenet', valid_patch_rate=0.75,
                             augment=True, vertical_flip_rate=0.4, horizontal_flip_rate=0.4, rotation_rate=0.4, hue_factor=0.2, test_flag=False):
        train_dir = '/projects/bcxi/nathanj/commonPatchData/'
        mapsh5i, map_names = PatchDataGenerator._load_maps_static(train_dir, patch_size, overlap)
        return DataLoader(PatchDataGenerator(map_names, mapsh5i, batch_size, patch_size, overlap, norm_type, valid_patch_rate,
                                             augment, vertical_flip_rate, horizontal_flip_rate, rotation_rate, hue_factor, test_flag), 
                          batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    @staticmethod
    def create_validation_dataset(batch_size=32, patch_size=256, overlap=15, norm_type='imagenet', valid_patch_rate=0.75,
                                  augment=True, vertical_flip_rate=0.4, horizontal_flip_rate=0.4, rotation_rate=0.4, hue_factor=0.2, test_flag=False):
        validation_dir = '/projects/bcxi/nathanj/commonPatchData/validation/'
        mapsh5i, map_names = PatchDataGenerator._load_maps_static(validation_dir, patch_size, overlap)
        return DataLoader(PatchDataGenerator(map_names, mapsh5i, batch_size, patch_size, overlap, norm_type, valid_patch_rate,
                                             augment, vertical_flip_rate, horizontal_flip_rate, rotation_rate, hue_factor, test_flag), 
                          batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    @staticmethod
    def _load_maps_static(h5_dir, patch_size, overlap):
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

        patch_sizes = [128, 256, 512, 1024]  # Supported patch sizes

        for size in patch_sizes:
            sub_dir = os.path.join(h5_dir, str(size), str(overlap))
            size_mapsh5i = {}
            files_name = [f for f in glob.glob(os.path.join(sub_dir, "*.hdf5")) if os.path.basename(f) not in exclude_files]
            for file in files_name:
                path = os.path.join(sub_dir, file)
                mapname = os.path.basename(path).replace(".hdf5", "")
                size_mapsh5i[mapname] = H5Image(path, "r")
            mapsh5i[size] = size_mapsh5i  

        map_names = list(mapsh5i[patch_size].keys())  # Default to input patch size for map_names
        return mapsh5i, map_names
