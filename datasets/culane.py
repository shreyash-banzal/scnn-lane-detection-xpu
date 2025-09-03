"""
CULane dataset implementation for SCNN training
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CULaneDataset(Dataset):
    """
    CULane dataset implementation for SCNN training
    """
    def __init__(self, data_root, phase='train', transform=None,
                 input_size=(288, 800)):
        self.data_root = data_root
        self.phase = phase
        self.transform = transform
        self.input_size = input_size

        # Load data list
        list_file = os.path.join(data_root, 'list', f'{phase}_gt.txt')
        self.data_list = self._load_data_list(list_file)
        print(f"Loaded {len(self.data_list)} {phase} samples")

    def _load_data_list(self, list_file):
        """Load data list from file"""
        data_list = []
        if not os.path.exists(list_file):
            print(f"Warning: {list_file} not found, using empty dataset")
            return data_list

        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    image_path = parts[0]
                    label_path = parts[1]
                    existence = [int(x) for x in parts[2:6]]  # 4 lanes
                    data_list.append({
                        'image': image_path,
                        'label': label_path,
                        'existence': existence
                    })
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_info = self.data_list[idx]

        # Load image, strip leading slash if present
        rel_image_path = data_info['image'].lstrip('/')
        image_path = os.path.join(self.data_root, rel_image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            # Return a dummy sample
            return self._get_dummy_sample()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load segmentation label, strip leading slash if present
        rel_label_path = data_info['label'].lstrip('/')
        label_path = os.path.join(self.data_root, rel_label_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            print(f"Warning: Could not load label {label_path}")
            label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Resize to input size
        image = cv2.resize(image, self.input_size)
        label = cv2.resize(label, self.input_size, interpolation=cv2.INTER_NEAREST)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label = torch.from_numpy(label).long()
        existence = torch.tensor(data_info['existence'], dtype=torch.float32)

        return {
            'image': image,
            'segmentation': label,
            'existence': existence,
            'image_path': data_info['image']
        }

    def _get_dummy_sample(self):
        """Return a dummy sample in case of loading errors"""
        image = torch.zeros(3, self.input_size[1], self.input_size[0])
        label = torch.zeros(self.input_size[1], self.input_size[0], dtype=torch.long)
        existence = torch.zeros(4, dtype=torch.float32)

        return {
            'image': image,
            'segmentation': label,
            'existence': existence,
            'image_path': 'dummy'
        }
