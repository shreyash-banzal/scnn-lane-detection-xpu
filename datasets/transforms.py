"""
Data transformations for lane detection
"""
import torch
from torchvision import transforms
import numpy as np

class ToTensor(object):
    """Convert numpy arrays to PyTorch tensors"""
    def __call__(self, image):
        # Convert HWC to CHW
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).float() / 255.0

class Normalize(object):
    """Normalize image with mean and std"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class RandomHorizontalFlip(object):
    """Random horizontal flip with probability p"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image):
        if np.random.random() < self.p:
            return np.fliplr(image).copy()
        return image

def get_transforms():
    """Get training and validation transforms"""
    train_transform = transforms.Compose([
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
