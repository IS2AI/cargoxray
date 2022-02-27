from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union
from numpy import double

import pandas as pd
from tqdm import tqdm
import random


def parse_region(region)\
        -> Optional[Tuple[int, int, int, int, Optional[str]]]:

    if region is None:
        return None

    # Try to indentify bbox shape
    if region['shape_attributes']['name'] == 'polyline'\
            or region['shape_attributes']['name'] == 'polygon':
        x_points = region['shape_attributes']['all_points_x']
        y_points = region['shape_attributes']['all_points_y']

    elif region['shape_attributes']['name'] == 'rect':
        x_points = region['shape_attributes']['x']
        y_points = region['shape_attributes']['y']
        w = region['shape_attributes']['width']
        h = region['shape_attributes']['height']

        x_points = [x_points, x_points + w, x_points + w, x_points]
        y_points = [y_points, y_points, y_points + h, y_points + h]
    else:
        return None

    x = min(x_points)
    w = max(x_points) - min(x_points)

    y = min(y_points)
    h = max(y_points) - min(y_points)

    try:
        label = region['region_attributes']['class name']
    except KeyError:
        label = None

    bbox = (x, y, w, h)

    return bbox, label


def load_or_create_frame(path, columns, index) -> pd.DataFrame:

    if not path.exists():
        frame = pd.DataFrame(columns=columns)
    else:
        try:
            frame = pd.read_json(path,
                                 orient='records',
                                 typ='frame',
                                 compression='gzip')
        except:
            frame = pd.read_json(path,
                                 orient='records',
                                 typ='frame')
    if frame.empty:
        frame = pd.DataFrame(columns=columns)

    frame = frame.set_index(index)

    return frame


def make_ref_cache(path) -> Dict[str, Path]:
    """Generates reference lookup table for faster search of files.
    Reference is concatenation of filename and its size. Build dictionary 
    reference to file path.

    Args:
        path (_type_): Root dir to build reference

    Returns:
        Dict[str, Path]: Reference to file path.
    """

    cache: Dict[str, Path] = {}

    f: Path
    for f in tqdm(list(path.glob('**/*')),
                  desc='Reference cache'):
        if f.is_file():
            cache['{}{}'.format(
                f.name,
                f.stat().st_size)] = f

    return cache


def fix_label(label: str,
              label_replacements: Dict[str, str]) -> str:

    if not isinstance(label, str):
        return None

    res = label.lower().strip().replace(' ', '_')

    try:
        res = label_replacements[res]
    except KeyError:
        pass

    return res


def load_label_replacements(path: Union[str, Path]) -> Dict[str, str]:

    label_replacements = pd.read_csv(path,
                                     names=['original', 'typos', 'merge']
                                     ).set_index('original')

    return label_replacements['typos'].to_dict()


def split(data: List,
          weights: List[float]) -> List:

    data = data.copy()
    weights = weights.copy()

    assert sum(weights) == 1

    for i in range(1, len(weights)):
        weights[i] += weights[i - 1]
    
    splits = [[] for i in range(len(weights))]

    for i in data:
        for bucket, w in enumerate(weights):
            if i % 100 <= w * 100:
                splits[bucket].append(i)
                break

    return splits


def convert_to_yolo(bbox, image_shape):

    x, y, w, h = bbox
    iw, ih = image_shape

    x_center = x + w/2
    y_center = y + h/2

    new_x = x_center / iw
    new_y = y_center / ih

    new_w = w / iw
    new_h = h / ih

    return new_x, new_y, new_w, new_h
