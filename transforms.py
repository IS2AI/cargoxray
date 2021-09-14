from typing import Tuple, Union
import torch
import torchvision
import random
from PIL import Image


class RandomHorizontalFlip:

    def __init__(self, p=0.5):
        self.p = p
        self.flipper = torchvision.transforms.RandomHorizontalFlip(1)

    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if random.random() <= self.p:

            image: torch.Tensor
            bboxes: torch.Tensor
            labels: torch.Tensor

            image, bboxes, labels = sample

            flipped_image = self.flipper(image)
            
            pivot = torch.Tensor((1, 0, 1, 0)) \
                .mul(image.size(2)) \
                .expand(bboxes.size(0), -1)
            flipped_bboxes = pivot - bboxes
            flipped_bboxes = flipped_bboxes.abs()

            return (flipped_image, flipped_bboxes, labels)

        else:
            return sample
