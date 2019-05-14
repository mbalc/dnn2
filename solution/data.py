import os
import random
from PIL import Image

import itertools

import torch
import numpy as np

import glob


import torchvision
from torchvision import datasets, transforms
from torch.utils import data as torchdata

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
np.random.seed(42)
random.seed(42)

DATASET_PATH = os.path.join(os.getcwd(), 'cityscapes')
COLOR_CLASS_MAP_PATH = os.path.join(os.getcwd(), 'colors')
ROOT_SUFFIXES = ["Training", "Test"]

BATCH_SIZE = 16
NUM_WORKERS = 32

CLASS_COUNT = 30
VALIDATION_SET_SIZE = 500

INPUT_IMG_SIZE = 256 # images are scaled to a square with this edge length (preventing a case where a image has different dimensions)

TRANSFORMATIONS = [
    transforms.functional.hflip,
] # future TODO - allow transformations that apply only to the input, i.e. contrast/saturation/brightness

    

class CityscapesDataset(torchdata.Dataset):
    def __init__(self, translist = transforms.Compose([])):
        self.transforms = translist

        self.load_images()
        self.load_color_mapper()
        # self.preprocess_images()

    def load_images(self):
        """Load images to memory"""
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

        self.all_image_paths = glob.glob(DATASET_PATH + '/*.png') 

    def load_color_mapper(self):
        """Either generate new color map from images, or load from previously generated save"""
        self.generate_color_class_map()

    # def preprocess_images():
    #     """Assign expected outputs their classes per pixel, according to color map"""

    def generate_color_class_map(self):
        self.pixel_values = set()
        for path in self.all_image_paths:
            img = self.img_from_path(path)
            (image, output) = self.split_input_image(img)
            output = self.pixels_to_class_codes(output)
            self.pixel_values.update(output.flatten().cpu().numpy())
            if (len(self.pixel_values) >= CLASS_COUNT): break ## we only loop in case we don't find all the classes on the first image

        self.color_map = dict()
        for idx, class_code in enumerate(self.pixel_values):
            self.color_map[class_code] = idx

    def img_from_path(self, path):
        img = Image.open(path)
        t = self.to_tensor(img)
        img.close()
        return t

    def transform(self, img):
        img = self.to_pil(img)
        img = self.transforms(img)
        return self.to_tensor(img)

    def pixels_to_class_codes(self, out):
        return ((out[0] * 255 * 256 * 256) + (out[1] * 255 * 256) + (out[2] * 255)).clone().long()

    def class_code_to_class(self, class_code):
        return self.color_map[class_code]

    def split_input_image(self, image):
        (img, out) = torch.split(image, INPUT_IMG_SIZE, dim=2)

        img = self.transform(img)
        out = self.transform(out)
        return (img, out)

    def __getitem__(self, idx):
        path = self.all_image_paths[idx]
        img = self.img_from_path(path)
        (img, out) = self.split_input_image(img)
        out = self.pixels_to_class_codes(out)
        out.apply_(self.class_code_to_class)

        return (img, out)

    def __len__(self):
        return len(self.all_image_paths)


def load_datasets():
    randomized_transformations = transforms.RandomOrder([
        transforms.RandomApply([tr])
        for tr in TRANSFORMATIONS
    ])


    all_transform_combinations = []
    for n in range(len(TRANSFORMATIONS) + 1): # (from 0 to len)
        for comb in itertools.permutations(TRANSFORMATIONS, n):
            all_transform_combinations.append(transforms.Compose(comb))

    train_dataset = CityscapesDataset(randomized_transformations)
    valid_datasets = [CityscapesDataset(translist) for translist in all_transform_combinations]
    ids = list(range(len(train_dataset)))

    random.shuffle(ids)

    train_sampler = torchdata.sampler.SubsetRandomSampler(ids[VALIDATION_SET_SIZE:])
    valid_sampler = torchdata.sampler.SubsetRandomSampler(ids[:VALIDATION_SET_SIZE])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    valid_loaders = [
        torch.utils.data.DataLoader(dset, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=NUM_WORKERS)
        for dset in valid_datasets
    ]

    train_size = len(train_sampler)
    valid_size = len(valid_sampler) * len(valid_datasets)

    return train_size, valid_size, train_loader, valid_loaders

    
    

