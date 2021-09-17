from typing import Tuple, Union
import torch
from torch import Tensor
import torchvision
import random
from PIL import Image
import albumentations as alb
import torchvision.transforms as tf


bbox_params = alb.BboxParams(format='pascal_voc',
                             label_fields=('labels',))

aug = alb.Compose(
    transforms=[

        alb.Affine(scale=1,
                   rotate=3,
                   shear=3,
                   mode=cv2.BORDER_REFLECT101,
                   p=0.25),

        alb.SmallestMaxSize(max_size=800),

        alb.RandomCrop(800, 800),

        alb.HorizontalFlip(p=0.25),

        alb.Equalize(p=0.25),

        alb.RandomGamma(gamma_limit=(85, 115),
                        p=0.25),

    ],
    bbox_params=bbox_params
)

wbboxes = tf.ToPILImage()(torchvision.utils.draw_bounding_boxes(
    torch.from_numpy(sample['image']).expand(3, -1, -1), 
    boxes=torch.IntTensor(sample['bboxes']),
    labels=sample['labels']
))
disp(wbboxes)
