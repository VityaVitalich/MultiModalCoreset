import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F


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


class DepthNormalizer:
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.float() / (2**15 - 1)
        return img


class MultiHorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img, **kwargs):
        if self.flip:
            return F.hflip(img)
        else:
            return img

    def set_param(self):
        if np.random.binomial(1, p=self.p):
            self.flip = True
        else:
            self.flip = False


class MultiRandomRotate:
    def __init__(self, p, angle):
        self.max_angle = angle
        self.p = p
        self.expand = False
        self.center = None
        self.interpolation = transforms.InterpolationMode.NEAREST
        self.fill_values = {"rgb": -5.4, "semseg": 255, "depth": 0}

    def __call__(self, img, modality, **kwargs):
        if self.rotate:
            fill_value = self.fill_values[modality]
            if modality == "semseg":
                img = img.unsqueeze(0)

            output = F.rotate(
                img,
                self.cur_angle,
                self.interpolation,
                self.expand,
                self.center,
                fill_value,
            )

            if modality == "semseg":
                return output.squeeze(0)

            return output
        else:
            return img

    def set_param(self):
        self.cur_angle = np.random.uniform(0, self.max_angle)
        if np.random.binomial(1, p=self.p):
            self.rotate = True
        else:
            self.rotate = False


class MultiVerticalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img, **kwargs):
        if self.flip:
            return F.vflip(img)
        else:
            return img

    def set_param(self):
        if np.random.binomial(1, p=self.p):
            self.flip = True
        else:
            self.flip = False
