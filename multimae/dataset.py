import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class FirstChannelTransform:
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor):
        return x[0]


class LongTransform:
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor):
        return x.long()


class Randomizer:
    def __init__(self, p, transform):
        self.p = p
        self.transform = transform

    def __call__(self, img):
        if np.random.binomial(1, p=self.p):
            return self.transform(img)
        return img


class MultiModalDataset(Dataset):
    def __init__(
        self,
        root_dir,
        train_transform=None,
        target_transofrm=None,
        input_tasks=["rgb"],
        output_task="depth",
    ):
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.target_transform = target_transofrm
        self.input_tasks = input_tasks
        self.output_task = output_task
        self.all_tasks = input_tasks + [output_task]
        for task_name in self.all_tasks:
            task_path = os.path.join(root_dir, task_name)
            self.__dict__[task_name + "_path"] = task_path
            self.__dict__[task_name + "_files"] = sorted(os.listdir(task_path))

    def __len__(self):
        task = self.all_tasks[0]
        return len(self.__dict__[task + "_files"])

    def __getitem__(self, idx):
        # obtain input object
        sample_dict = {}
        for task in self.input_tasks:
            file_name = self.__dict__[task + "_files"][idx]
            file_path = os.path.join(self.__dict__[task + "_path"], file_name)
            x = Image.open(file_path)
            if task == "rgb":
                x = x.convert("RGB")
            if task == "semantic":
                x = x.convert("L")

            if self.train_transform:
                x = self.train_transform[task](x)
            sample_dict[task] = x

        # obtain target
        target_file_name = self.__dict__[self.output_task + "_files"][idx]
        target_file_path = os.path.join(
            self.__dict__[self.output_task + "_path"], file_name
        )
        y = Image.open(target_file_path)

        assert file_name == target_file_name

        if self.target_transform:
            y = self.target_transform(y)

        return sample_dict, y
