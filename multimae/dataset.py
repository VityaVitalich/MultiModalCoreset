import os
from PIL import Image
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(
        self,
        root_dir,
        train_transform=None,
        target_transofrm=None,
        multimodal_augmentations=None,
        input_tasks=["rgb"],
        output_task="depth",
        training=True,
    ):
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.target_transform = target_transofrm
        self.multimodal_augmentations = multimodal_augmentations
        self.input_tasks = input_tasks
        self.output_task = output_task
        self.training = training
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

        if self.multimodal_augmentations and self.training:
            sample_dict, y = self.make_augmentations(sample_dict, y)

        return sample_dict, y

    def make_augmentations(self, sample_dict, y):
        for augmentation in self.multimodal_augmentations:
            # print(augmentation)
            augmentation.set_param()
            for task, img in sample_dict.items():
                sample_dict[task] = augmentation(img, modality=task)
            y = augmentation(y, modality=self.output_task)
            # print('completed')

        return sample_dict, y
