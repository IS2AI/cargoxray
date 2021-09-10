import torch.utils.data
from pathlib import Path
import json
from typing import Union
import pandas as pd
from PIL import Image
import numpy as np


class CargoXRay(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir: Union[str, Path],
                 transform=None):

        data_dir = Path(data_dir)

        self.annotations = pd.read_json(data_dir / 'annotations.json')
        self.images = pd.read_json(data_dir / 'images.json')

        self.annotations = self.annotations.loc[
            self.annotations['label'].notnull()]

        self.annotations = self.annotations.merge(
            self.images, 'inner',
            left_on='image_id',
            right_on='id')

        self.annotations = self.annotations.drop(
            ['id_y', 'image_id'], axis='columns')
        self.annotations = self.annotations.rename(
            {'id_x': 'id'}, axis='columns')
        self.annotations = self.annotations.set_index('id')

        self.labels: pd.Series = self.annotations['label'] \
            .drop_duplicates() \
            .sort_values() \
            .reset_index(drop=True)
        self.labels = self.labels.to_list()
        self.labels = {val: idx for idx, val in enumerate(self.labels)}


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sel = self.annotations.iloc[idx]

        image = Image.open(sel['filepath']).convert('L')

        x1 = min(sel['x_points'])
        y1 = min(sel['y_points'])
        x2 = max(sel['x_points'])
        y2 = max(sel['y_points'])

        x = (x1 + x2) // 2
        y = (y1 + y2) // 2

        w = x2 - x1
        h = y2 - y1

        x /= image.width
        y /= image.height
        w /= image.width
        h /= image.height

        label = self.labels[sel['label']]

        return (sel['filepath'], (label, x, y, w, h,))

    def __len__(self):
        return len(self.annotations)


if __name__ == '__main__':
    ds = CargoXRay('data')
    
    for img, ann in ds:
        print(img)
