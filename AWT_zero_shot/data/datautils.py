import os
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *

ID_to_DIRNAME={
    'imagenet': 'imagenet',
    'imagenet_a': 'imagenet-adversarial',
    'imagenet_sketch': 'imagenet-sketch',
    'imagenet_r': 'imagenet-rendition',
    'imagenetv2': 'imagenetv2',
    'oxford_flowers': 'oxford_flowers',
    'dtd': 'dtd',
    'oxford_pets': 'oxford_pets',
    'stanford_cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'fgvc_aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
    'caltech256': 'caltech256',
    'cub': 'cub',
    'birdsnap': 'birdsnap',
}

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id in ['imagenet', 'imagenet_a', 'imagenet_sketch', 'imagenet_r', 'imagenetv2']:
        if set_id == 'imagenet':
            testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'images', 'val')
        elif set_id == 'imagenetv2':
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenetv2-matched-frequency-format-val')
        elif set_id == 'imagenet_a':
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenet-a')
        elif set_id == 'imagenet_r':
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenet-r')
        elif set_id == 'imagenet_sketch':
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'images')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    else:
        raise NotImplementedError
        
    return testset


# Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def aug(image, preprocess):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    return x_processed


class Augmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [aug(x, self.preprocess) for _ in range(self.n_views)]
        return [image] + views
