import os
import random
from PIL import Image

import torch

import glob


import torchvision
from torchvision import datasets, transforms
from torch.utils import data as torchdata


DATASET_PATH = os.path.join(os.getcwd(), 'cityscapes')
COLOR_CLASS_MAP_PATH = os.path.join(os.getcwd(), 'colors')
ROOT_SUFFIXES = ["Training", "Test"]

BATCH_SIZE = 92
NUM_WORKERS = 32

CLASS_COUNT = 30
VALIDATION_SET_SIZE = 500

INPUT_IMG_SIZE = 256 # images are scaled to a square with this edge length (preventing a case where a image has different dimensions)

    

class CityscapesDataset(torchdata.Dataset):
    def __init__(self):
        self.load_images()
        self.load_color_mapper()
        # self.preprocess_images()

    def load_images(self):
        """Load images to memory"""
        print('Loading image paths...')
        self.to_tensor = transforms.ToTensor()
        self.all_image_paths = glob.glob(DATASET_PATH + '/*.png') 

    def load_color_mapper(self):
        """Either generate new color map from images, or load from previously generated save"""
        print('Loading color mapper...')
        self.generate_color_class_map()

    # def preprocess_images():
    #     """Assign expected outputs their classes per pixel, according to color map"""
    #     print('Preprocessing images...')

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

    def pixels_to_class_codes(self, out):
        return ((out[0] * 255 * 256 * 256) + (out[1] * 255 * 256) + (out[2] * 255)).clone().int()

    def split_input_image(self, image):
        return torch.split(image, INPUT_IMG_SIZE, dim=2)

    def __getitem__(self, idx):
        path = self.all_image_paths[idx]
        img = self.img_from_path(self)
        (img, out) = self.split_input_image(img)
        out = pixels_to_class_codes(out)

        img.to('cuda')
        return self.images[idx]

    def __len__(self):
        return len(self.all_image_paths)


def load_datasets():
    image_dataset = CityscapesDataset()
    ids = list(range(len(image_dataset)))

    random.seed(42)
    random.shuffle(ids)

    train_sampler = torchdata.sampler.SubsetRandomSampler(ids[VALIDATION_SET_SIZE:])
    valid_sampler = torchdata.sampler.SubsetRandomSampler(ids[:VALIDATION_SET_SIZE])
    
    train_loader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=NUM_WORKERS)

    return train_loader, valid_loader

    
    

