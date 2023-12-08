import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.rgb_path = os.path.join(root_dir, 'rgb')
        self.depth_path = os.path.join(root_dir, 'depth_euclidean')
        self.rgb_files = sorted(os.listdir(self.rgb_path))
        self.depth_files = sorted(os.listdir(self.depth_path))

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_name = self.rgb_files[idx]
        depth_name = self.depth_files[idx]

        assert rgb_name == depth_name

        rgb_img_path = os.path.join(self.rgb_path, rgb_name)
        depth_img_path = os.path.join(self.depth_path, depth_name)

        rgb_image = Image.open(rgb_img_path).convert('RGB')
        depth_image = Image.open(depth_img_path)#.convert('L')  # 'L' mode for grayscale images

        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)

        return rgb_image, depth_image


