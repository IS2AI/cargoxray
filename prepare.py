import json
from pathlib import Path
import pathlib
from hashlib import md5
from typing import Any, Dict, Union
import tqdm
import pandas as pd
import shutil

IMPORT_DIR = Path('import')
DESTINATION_DIR = Path('test')


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

    try:
        images = pd.read_json(outdir.joinpath('images.json'),
                              orient='records',
                              typ='frame')
        next_id = images['id'].max() + 1
    except:
        images = pd.DataFrame(columns=[
            'id',
            'filepath',
            'size',
            'md5',
            'old_filepath'
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
            'old_filepath': src.as_posix()
        })

        next_id += 1

    images = images.append(new_images, ignore_index=True)
    images = images.drop_duplicates(['size', 'md5', 'old_filepath'])

    no_md5_images = images.loc[images['md5'].isna()]
    images = images.loc[images['md5'].notna()]

    images = images \
        .drop('old_filepath',
              axis=1) \
        .groupby('md5') \
        .first() \
        .merge(images[['md5', 'old_filepath']],
               on='md5')

    images = images.append(no_md5_images, ignore_index=True)

    images = images[['id', 'filepath', 'size',
                     'md5', 'old_filepath']].sort_values('id')

    return images


def prepare_annotations(dir: Union[str, Path],
                        outdir,
                        images: pd.DataFrame):

    dir = Path(dir)

    try:
        annotations = pd.read_json(outdir.joinpath('annotations.json'),
                                   orient='records',
                                   typ='frame')
        next_id = annotations['id'].max()
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

    src: Path
    for src in tqdm.tqdm(list(dir.glob('**/*.json')),
                         desc='Processing JSONs'):

        with src.open('rb') as fs:
            anns = json.load(fs)

        ann: Dict[str, Any]
        for ann in anns.values():

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


                img_id = images \
                    .loc[(images['old_filepath'].str.endswith(ann['filename'])) \
                    & (images['size'] == ann['size'])]
                

                if img_id.empty:

                    img_id = pd.DataFrame({
                        'id': next_image_id,
                        'filepath': None,
                        'size': ann['size'],
                        'md5': None,
                        'old_filepath': ann['filename'],
                    },index=(1,))
                    next_image_id += 1

                    images = images.append(img_id, ignore_index=True)

                img_id = img_id.iloc[0]['id']

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

    return images, annotations


def run():
    DESTINATION_DIR.mkdir(exist_ok=True)
    DESTINATION_DIR.joinpath('images').mkdir(exist_ok=True)

    images = prepare_images(IMPORT_DIR, DESTINATION_DIR)
    images, annotations = prepare_annotations(
        IMPORT_DIR, DESTINATION_DIR, images)

    images.to_json(DESTINATION_DIR / 'images.json',
                   orient='records',
                   indent=2)
    annotations.to_json(DESTINATION_DIR / 'annotations.json',
                        orient='records',
                        indent=2)

    breakpoint()

    for idx, row in tqdm.tqdm(images.loc[images['filepath'].notna()].iterrows(),
                              desc='Copying images',
                              total=len(images.loc[images['filepath'].notna()])):
        if not Path(row['filepath']).exists():
            shutil.copy(row['old_filepath'], row['filepath'])


if __name__ == '__main__':
    run()
