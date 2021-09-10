import pandas as pd
from pathlib import Path
from cargoxray import CargoXRay
import torch
import torch.utils
import torch.utils.data
import torchvision


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

    sel: pd.DataFrame = ds.annotations.merge(
        sel, on='label')

    weights = []

    for idx, ann in sel.iterrows():

        if ann['count'] > 1000:
            weights.append(1 / ann['count'])
        else:
            weights.append(0)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=10000)

    dl = torch.utils.data.DataLoader(ds, sampler=sampler, num_workers=8)

    stat = {}
    for fp, ann in dl:
        if ann[0].item() not in stat:
            stat[ann[0].item()] = 0
        stat[ann[0].item()] += 1

    print(stat)


if __name__ == '__main__':
    run('data')
