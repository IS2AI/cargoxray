import json
from pathlib import Path
import pathlib
from hashlib import md5
from typing import Any, Dict, Union
import tqdm
import pandas as pd
import shutil
import multiprocessing as mp

IMPORT_DIR = Path('import')
DESTINATION_DIR = Path('data')


def cglob(path, patterns):
    for pattern in patterns:
        for p in path.glob(pattern):
            yield p


def prepare_images(dir: Union[str, Path], outdir: Union[str, Path]) -> Dict[str, str]:
    """Scans directory for .tif and .jpg images and adds them the database
    of images

    Args:
        dir (str | Path): Input folder
        outdir (str | Path): Destination folder where to copy images

    Returns:
        Dict[str, str]: A dictionary of filesnames lookup - where a file had been copied
    """

    dir = Path(dir)

    # Try to load existing database, create new otherwise
    try:
        images = pd.read_json(outdir.joinpath('images.json'),
                              orient='records',
                              typ='frame')

        images = images.assign(old_filepath=len(images)*[None],
                               ref=len(images)*[None])

        next_id = images['id'].max() + 1
    except:
        images = pd.DataFrame(columns=[
            'id',
            'filepath',
            'size',
            'md5',
            'old_filepath',
            'ref',
        ])
        next_id = 1

    new_images = []

    src: Path
    for src in tqdm.tqdm(list(cglob(dir, ('**/*.tif', '**/*.jpg'))),
                         desc='Images hashing'):

        dst = outdir.joinpath(
            f'images/{next_id:0>5X}{src.suffix}').as_posix()

        new_images.append({
            'id': next_id,
            'filepath': dst,
            'size': src.stat().st_size,
            'md5': md5(src.read_bytes()).hexdigest(),
            'old_filepath': src.as_posix(),
            'ref': '{}{}'.format(src.name, src.stat().st_size),
        })

        next_id += 1

    images = images.append(new_images, ignore_index=True)

    images = images \
        .drop(['old_filepath', 'ref'],
              axis=1) \
        .sort_values('id') \
        .groupby('md5') \
        .first() \
        .merge(images[['md5', 'old_filepath', 'ref']],
               on='md5')

    return images


def prepare_annotations(dir: Union[str, Path],
                        outdir,
                        images: pd.DataFrame):

    dir = Path(dir)

    try:
        annotations = pd.read_json(outdir.joinpath('annotations.json'),
                                   orient='records',
                                   typ='frame')
        next_id = annotations['id'].max() + 1
    except:
        annotations = pd.DataFrame(None,
                                   columns=(
                                       'id',
                                       'image_id',
                                       'x_points',
                                       'y_points',
                                       'label'
                                   ))
        next_id = 1

    next_image_id = images['id'].max() + 1

    new_annotations = []
    bad_annotations = []

    src: Path
    for src in tqdm.tqdm(list(dir.glob('**/*.json')),
                         desc='Processing JSONs'):

        with src.open('rb') as fs:
            anns = json.load(fs)

        ann: Dict[str, Any]
        for ref, ann in anns.items():

            if isinstance(ann['regions'], dict):
                regions = list(ann['regions'].values())
            else:
                regions = ann['regions']

            for reg in regions:

                if reg is None:
                    continue

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
                    bad_annotations.append({
                        'img_filename': ann['filename'],
                        'img_filesize': ann['size'],
                        'region': reg,
                        'reason': 'Undefined shape'
                    })
                    continue

                try:
                    label = reg['region_attributes']['class name']
                except KeyError:
                    label = None

                try:
                    sel = images.loc[images['ref'] == ref]
                    img_id = sel.iloc[0]['id']
                    if len(sel['md5'].unique()) > 1:
                        print(sel)
                        continue
                except:
                    bad_annotations.append({
                        'img_filename': ann['filename'],
                        'img_filesize': ann['size'],
                        'region': reg,
                        'reason': 'Missing image'
                    })
                    continue

                new_ann = {
                    'id': next_id,
                    'image_id': img_id,
                    'x_points': x,
                    'y_points': y,
                    'label': label
                }

                next_id += 1

                new_annotations.append(new_ann)

    annotations = annotations.append(new_annotations, ignore_index=True)

    annotations['x_points'] = annotations['x_points'].apply(tuple)
    annotations['y_points'] = annotations['y_points'].apply(tuple)

    annotations = annotations.drop_duplicates((
        'image_id',
        'x_points',
        'y_points',
        'label',))

    return annotations


def copy_images(images: pd.DataFrame):

    DESTINATION_DIR.joinpath('images').mkdir(parents=True, exist_ok=True)

    for idx, row in images.loc[images['old_filepath'].notna()].iterrows():
        src = Path(row['old_filepath'])
        dst = Path(row['filepath'])

        if not dst.exists():
            shutil.copy(src, dst)


def clean_images(images: pd.DataFrame):

    images = images.drop(['old_filepath', 'ref'], axis=1)
    images = images.drop_duplicates(subset=['filepath'])

    return images


def write_json(outdir, images: pd.DataFrame, annotations: pd.DataFrame):

    DESTINATION_DIR.mkdir(exist_ok=True)

    images.to_json(DESTINATION_DIR / 'images.json',
                   orient='records',
                   indent=2)
    annotations.to_json(DESTINATION_DIR / 'annotations.json',
                        orient='records',
                        indent=2)


def run():

    images = prepare_images(IMPORT_DIR, DESTINATION_DIR)

    annotations = prepare_annotations(
        IMPORT_DIR, DESTINATION_DIR, images)

    copy_images(images)

    images = clean_images(images)

    write_json(DESTINATION_DIR, images, annotations)


if __name__ == '__main__':
    run()
