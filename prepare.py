import json
from pathlib import Path
from hashlib import md5
from typing import Any, Dict, Union
import tqdm
import pandas as pd
import shutil

IMPORT_DIR = Path('/Users/contactone/Desktop/raw_x_ray backup')
DESTINATION_DIR = Path('data')


def cglob(path, patterns):
    for pattern in patterns:
        for p in path.glob(pattern):
            yield p


def prepare_images(dir, outdir):

    dir = Path(dir)

    try:
        images = pd.read_json(outdir.joinpath('images.json.gz'),
                              orient='records',
                              compression='gzip',
                              typ='frame')
        next_id = images['id'].max() + 1
    except:
        images = pd.DataFrame(columns=[
            'id', 'filepath', 'size', 'md5'])
        next_id = 1

    auxilary = {}

    if next_id is None:
        next_id = images['id'].max(numeric_only=True) + 1

    src: Path
    for src in tqdm.tqdm(list(cglob(dir, ('**/*.tif', '**/*.jpg')))):

        md5_hash = md5(src.read_bytes()).hexdigest()

        dst = outdir.joinpath(
            f'images/{next_id:0>5X}{src.suffix}').as_posix()

        dup = images.loc[images['md5'] == md5_hash]

        if not dup.empty:
            auxilary[f'{src.name}{src.stat().st_size}'] = dup['id'].iloc[0]
        else:
            auxilary[f'{src.name}{src.stat().st_size}'] = next_id

            images = images.append({
                'id': next_id,
                'filepath': dst,
                'size': src.stat().st_size,
                'md5': md5_hash
            }, ignore_index=True)

            next_id += 1

            shutil.copy2(src, dst)

    images.to_json(outdir.joinpath('images.json.gz'),
                   orient='records',
                   compression='gzip')

    return auxilary


def prepare_annotations(dir: Union[str, Path],
                        outdir,
                        auxilary: Dict[str, int]):

    dir = Path(dir)

    try:
        annotations = pd.read_json(outdir.joinpath('annotations.json.gz'),
                                   orient='records',
                                   compression='gzip',
                                   typ='frame')
        next_id = annotations['id'].max()
    except:
        annotations = pd.DataFrame(None,
                                   columns=(
                                       'id',
                                       'image_id',
                                       'x_points',
                                       'y_points',
                                       'label',
                                       'missing_image_name',
                                       'missing_image_size',
                                   ))
        annotations['image_id'] = annotations['image_id'].astype(int)
        next_id = 1

    new_annotations = []

    src: Path
    for src in tqdm.tqdm(list(dir.glob('**/*.json'))):

        with src.open('rb') as fs:
            anns = json.load(fs)

        ann: Dict[str, Any]
        for ref, ann in anns.items():

            if isinstance(ann['regions'], dict):
                regions = list(ann['regions'].values())
            else:
                regions = ann['regions']

            for reg in regions:

                if reg['shape_attributes']['name'] == 'polyline' \
                        or reg['shape_attributes']['name'] == 'polygon':
                    x = reg['shape_attributes']['all_points_x']
                    y = reg['shape_attributes']['all_points_y']

                elif reg['shape_attributes']['name'] == 'rect':
                    x = reg['shape_attributes']['x']
                    y = reg['shape_attributes']['y']
                    w = reg['shape_attributes']['width']
                    h = reg['shape_attributes']['height']

                    x = [x, x + w, x + w, x]
                    y = [y, y, y + h, y + h]
                else:
                    print('Omitting', reg['shape_attributes'])
                    continue

                try:
                    label = reg['region_attributes']['class name']
                except KeyError:
                    label = None

                new_ann = {
                    'id': next_id,
                    'image_id': auxilary.get(ref),
                    'x_points': x,
                    'y_points': y,
                    'label': label,
                    'missing_image_name': None,
                    'missing_image_size': None,
                }

                next_id += 1

                if new_ann['image_id'] is None:
                    new_ann['missing_image_name'] = ann['filename']
                    new_ann['missing_image_size'] = ann['size']

                new_annotations.append(new_ann)

    annotations = annotations.append(new_annotations, ignore_index=True)
    annotations['x_points'] = annotations['x_points'].apply(tuple)
    annotations['y_points'] = annotations['y_points'].apply(tuple)

    annotations = annotations.drop_duplicates(('image_id',
                                               'x_points',
                                               'y_points',
                                               'label',
                                               'missing_image_name',
                                               'missing_image_size'
                                               ))
    annotations.to_json(outdir.joinpath('annotations.json.gz'),
                        orient='records',
                        compression='gzip')


def run():
    DESTINATION_DIR.mkdir(exist_ok=True)
    DESTINATION_DIR.joinpath('images').mkdir(exist_ok=True)

    auxilary = prepare_images(IMPORT_DIR, DESTINATION_DIR)
    prepare_annotations(IMPORT_DIR, DESTINATION_DIR, auxilary)


if __name__ == '__main__':
    run()
