import pandas as pd
from pathlib import Path
from cargoxray import CargoXRay
import torch
import torch.utils
import torch.utils.data
import torchvision
import random

from typing import List, Dict
import shutil


def run(data_dir):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.AutoAugment(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(40),
        torchvision.transforms.RandomAutocontrast(),
        torchvision.transforms.Resize(640),
    ])

    ds = CargoXRay(data_dir)

    sel = ds.annotations.groupby('label').count()[['x_points']].rename(
        {'x_points': 'count'}, axis='columns')

    sel: pd.DataFrame = ds.annotations.reset_index().merge(
        sel, on='label').sort_values('id').set_index('id')

    print(sel)

    weights: List[float] = []

    for idx, ann in sel.iterrows():

        if ann['count'] > 500:
            weights.append(1. / ann['count'])

        else:
            weights.append(0)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=10000)

    dl = torch.utils.data.DataLoader(ds, sampler=sampler, num_workers=24)

    labels = {}
    train = set()

    for fp, ann in dl:

        label = ann[0][0]

        Path().read_text()
        Path().write_text()

        if label not in labels:
            labels[label] = len(labels)

        label = labels[label]

        if random.random() >= 0.1 or fp[0] in train:
            train.add(fp[0])
            if not Path('yolo/images/train', fp[0]).exists():
                shutil.copy(fp[0], 'yolo/images/train/')
            with Path(f'yolo/labels/train/{Path(fp[0]).stem}.txt').open('a') as f:
                f.write(str(label) + ' ')
                f.write(' '.join([str(x.item()) for x in ann[1:]]))
                f.write('\n')
        else:
            if not Path('yolo/images/val', fp[0]).exists():
                shutil.copy(fp[0], 'yolo/images/val/')
            with Path(f'yolo/labels/val/{Path(fp[0]).stem}.txt').open('a') as f:
                f.write(str(label) + ' ')
                f.write(' '.join([str(x.item()) for x in ann[1:]]))
                f.write('\n')

    with Path('yolo/yolo.yaml').open('w') as f:
        f.write('train: yolo/images/train\n')
        f.write('val: yolo/images/val\n')
        f.write(f'nc: {len(labels)}\n')
        f.write(f'names: [{", ".join(list(labels.keys()))}]\n')


def run2(data_dir):
    ds = CargoXRay(data_dir)
    
    ds.annotations.groupby('label').count()['x_points'].sort_values(ascending=False).to_csv('info.csv')
    print(len(ds.annotations))



if __name__ == '__main__':
    run2('data')
