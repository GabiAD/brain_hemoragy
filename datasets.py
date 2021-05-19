import os
from torch.utils.data import Dataset
import glob
from skimage import io
import torch
import numpy as np

class BrainsDataset(Dataset):
    def __init__(self, root, image_label_pairs):
        self.images_paths = [os.path.join(root, f"{ln.split(',')[0]}.png") for ln in image_label_pairs]
        self.images_labels = [ln.split(",")[1] for ln in image_label_pairs]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images_paths[idx]
        image = io.imread(img_name)
        label = int(self.images_labels[idx])

        return image, label
