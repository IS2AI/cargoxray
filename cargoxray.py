import torch.utils.data
from pathlib import Path
import json
from typing import Union

class CargoXRay(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir: Union[str, Path],
                 transform=None):
        with Path(data_dir) \
                .joinpath('annotations.json') \
                .open('r') as fs:
            self.annotations = json.load(fs)
        
        annotations = []
        for img in self.annotations:
            for ann in img['regions']:
                annotations.append({
                    'image': img['image'],
                    'region': ann,
                })
        self.annotations = annotations

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        

    def __len__(self):
        return len(self.annotation)


if __name__ == '__main__':
    ds = CargoXRay()
