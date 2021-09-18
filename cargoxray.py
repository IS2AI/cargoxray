import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from PIL import Image


class CargoXRay(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir: Union[str, Path] = 'data',
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

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[
            Image.Image,
            List[
                Tuple[
                    int,
                    int,
                    int,
                    int
                ]],
            List[str]]:

        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        sel_img = self.images.iloc[idx]

        image = Image.open(sel_img['filepath'])
        if image.mode == 'I;16':
            image = image.convert('I').point(
                [i/256 for i in range(2**16)], 'L')
        else:
            image = image.convert('L')
        # image = skimage.io.imread(sel_img['filepath'], as_gray=True)
        # image = torch.Tensor(image)
        # if image.max() <= 2**8:
        #     image.div(2**8)
        # elif image.max() <= 2**16:
        #     image.div(2**16)
        # image = image[None, :]

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

        # bboxes = torch.Tensor(bboxes)
        # labels = torch.Tensor(
        #     [self.labels.loc[self.labels['label'] == l].iloc[0].name for l in labels])

        return (image, bboxes, labels)


if __name__ == '__main__':
    ds = CargoXRay('data')
    print(len(ds))

    print(ds[9777])
