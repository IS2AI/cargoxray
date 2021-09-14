from skimage.util.dtype import convert
import torch.utils.data
from pathlib import Path
import json
from typing import Union
import pandas as pd
from PIL import Image
import numpy as np
import torch
import skimage.io


class CargoXRay(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir: Union[str, Path],
                 dropna=True,
                 transform=None):

        # Load dataframes
        data_dir = Path(data_dir)

        self.annotations: pd.DataFrame = pd.read_json(
            data_dir / 'annotations.json')
        self.images = pd.read_json(data_dir / 'images.json')

        # Drop bad annotations (missing labels)

        if dropna:
            self.annotations = self.annotations.dropna()

        # Drop images that do not have corresponding annotations

        self.images = self.annotations['image_id'].to_frame() \
            .drop_duplicates() \
            .merge(right=self.images,
                   how='inner',
                   on='image_id') \
            .set_index('image_id') \
            .sort_index()

        # Set annotations index

        self.annotations = self.annotations \
            .set_index('id') \
            .sort_index()

        # Generate datafram of labels
        # Labels id are assigned according to their frequency
        # i.e., most frequent label will have lower label_id

        self.labels = self.annotations \
            .groupby('label') \
            .count()['image_id'] \
            .to_frame() \
            .sort_values('image_id', ascending=False) \
            .drop(columns='image_id') \
            .reset_index()
        self.labels.index.rename('label_id', inplace=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        sel_img = self.images.iloc[idx]

        # image = Image.open(sel_img['filepath'])
        image = skimage.io.imread(sel_img['filepath'], as_gray=True)
        image = torch.Tensor(image)
        if image.max() <= 2**8:
            image.div(2**8)
        elif image.max() <= 2**16:
            image.div(2**16)
        image = image[None, :]

        sel_ann = self.annotations \
            .loc[self.annotations['image_id'] == sel_img.name]

        bboxes = []
        labels = []

        for _, ann in sel_ann.iterrows():
            bboxes.append((
                min(ann['x_points']),
                min(ann['y_points']),
                max(ann['x_points']),
                max(ann['y_points']),
            ))
            labels.append(ann['label'])

        bboxes = torch.Tensor(bboxes)
        labels = torch.Tensor([self.labels.loc[self.labels['label'] == l].iloc[0].name for l in labels])

        return (image, bboxes, labels)


if __name__ == '__main__':
    ds = CargoXRay('data')
    print(len(ds))

    print(ds[9777])
