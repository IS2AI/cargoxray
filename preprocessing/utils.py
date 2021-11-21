from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm


def parse_region(region) \
        -> Optional[Tuple[int, int, int, int, Optional[str]]]:

    if region is None:
        return None

    # Try to indentify bbox shape
    if region['shape_attributes']['name'] == 'polyline' \
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
