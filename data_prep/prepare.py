import json
from pathlib import Path
from hashlib import md5
from typing import Any, Dict, Union
import tqdm
import pandas as pd
import numpy as np


IMPORT_DIR = Path('data/import')


def compute_hashes() -> Dict[str, str]:
    """Compute MD5 hash strings for each file in the directory

    Returns:
        Dict[str, str]: Dictionary with structure "filepath": "md5 hash"
    """

    hashes: Dict[Path, bytes] = {}

    for path in tqdm.tqdm(list(IMPORT_DIR.glob('**/*')), 'Computing hashes'):
        if path.is_dir():
            with path.open('rb') as fs:
                hashes[path.as_posix()] = md5(fs.read()).hexdigest()

    return hashes


def load_annotations(annotations_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads annotation file into memory

    Args:
        annotations_path (Union[str, Path]): Path to JSON file with annotations

    Returns:
        Dict[str, Any]: The loaded annotations
    """

    print('Loading annotation file')

    annotations_path = Path(annotations_path)

    try:
        with annotations_path.open('r') as fs:
            annotations = json.load(fs)
    except FileNotFoundError:
        annotations = {
            'images': [],
            'annotations': [],
        }
    
    print('Loading annotation file... OK')

    return annotations


def run():

    annotations = load_annotations('data/annotations.json')

    images = []

    for ann in annotations:
        images.append(ann['image'])
    
    images = pd.DataFrame(images).drop_duplicates('filepath')
    
    print(images)

    return

    for img in annotations['images']:
        hashes[img['md5']] = img['filepath']
        counter = max(counter, img['id'])

    new_hashes = compute_hashes()

    copy_dst = {}

    for pth, hsh in new_hashes.items():
        if hsh in hashes:
            with Path(pth).open('rb') as fs1, \
                    Path(hashes[hsh]).open('rb') as fs2:
                if fs1.read() == fs2.read():
                    copy_dst[pth] = hashes[hsh]
                else:
                    copy_dst[pth]


if __name__ == '__main__':
    run()
