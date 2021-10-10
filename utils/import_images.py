
# # Constants & settings

import json
import logging
import multiprocessing as mp
import os
import pathlib
import shutil
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import tqdm
from IPython.display import display
from PIL import Image, UnidentifiedImageError

WORKDIR = '/raid/ruslan_bazhenov/projects/xray/cargoxray'
DATA_DIR = 'data'


def scan_images(images: pd.DataFrame]):
    # # Import images
    # ## Load existsing images databases if exist

    images = images.copy()

    try:
        images = pd.read_json(DATA_DIR / 'images2.json.gz',
                              orient='records',
                              typ='frame',
                              compression='gzip')
        images = images.set_index('image_id')

        next_id = images.index.max() + 1
    except:
        images = pd.DataFrame()
        images.index.name = 'image_id'
        next_id = 0

    # ## Scan & import image files

    image_path: Path

    for image_path in tqdm.tqdm(list(DATA_DIR.glob('images/**/*.tif')), desc='Images'):

        # Skip already imported images
        if len(images.loc[images['filepath'] == image_path
                          .relative_to(DATA_DIR)
                          .as_posix()]) > 0:
            continue

        try:
            image = Image.open(image_path)
        except UnidentifiedImageError:
            print('Broken image {}'.format(image_path))
            continue

        image_info = pd.Series(name=next_id, data={

            'md5': md5(image_path.read_bytes()).hexdigest(),
            'size': image_path.stat().st_size,
            'width': image.width,
            'height': image.height,
            'filepath': image_path.relative_to(DATA_DIR).as_posix(),

        })

        next_id += 1

        image.close()

        images = images.append(image_info)

    # ## Map duplicate images to the same location

    images = images\
        .drop(columns=['filepath'])\
        .drop_duplicates('md5')\
        .reset_index()\
        .merge(images[['md5',
                       'filepath']],
               on='md5') \
        .set_index('image_id') \
        .drop_duplicates()

    return images 
    # # Import annotations
    # ## Load existing annotations files


def run():
    global WORKDIR
    global IMAGES_DIR
    global DATA_DIR

    # # Change workdir and import libraries

    os.chdir(WORKDIR)
    DATA_DIR = Path(DATA_DIR)

    # # Preview files counts

    logging.info('Images dir content')

    sufcnt = {}
    for f in DATA_DIR.glob('images/**/*'):
        if f.is_file():
            try:
                sufcnt[f.suffix] += 1
            except KeyError:
                sufcnt[f.suffix] = 1

    for k, v in sufcnt.items():
        print(f'{k.strip("."):>10} : {v}')


    try:
        annotations = pd.read_json(DATA_DIR / 'annotations.json.gz',
                                   orient='records',
                                   typ='frame',
                                   compression='gzip')
        annotations = annotations.set_index('bbox_id')

        next_id = annotations.index.max() + 1

    except:

        annotations = pd.DataFrame()
        annotations.index.name = 'bbox_id'
        next_id = 0

    # ## Scan & import JSON files

    images['filename'] = images['filepath'].apply(lambda x: Path(x).name)

    for src in tqdm.tqdm(list(DATA_DIR.glob('images/**/*.json')),
                         desc='JSONs'):

        # Annotations loaded
        with src.open('rb') as fs:
            anns = json.load(fs)

        # For each group of annotations for one image
        ann: Dict[str, Any]
        for ref, ann in anns.items():

            # Try to find the corresponding image
            sel = images.loc[(images['filename'] == ann['filename'])
                             & (images['size'] == ann['size'])]

            sel = sel.drop_duplicates('md5')

            if len(sel) == 0:
                continue
            elif len(sel) > 1:
                display(sel)

            # In some JSON files regions are dict, in some are lists
            if isinstance(ann['regions'], dict):
                regions = list(ann['regions'].values())
            else:
                regions = ann['regions']

            # For each bbox
            for reg in regions:

                # Skip missing bboxes
                if reg is None:
                    continue

                # Try to indentify bbox shape
                if reg['shape_attributes']['name'] == 'polyline' or reg['shape_attributes']['name'] == 'polygon':
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

                x_center = (max(x) + min(x)) / 2 / sel.iloc[0]['width']
                w = (max(x) - min(x)) / sel.iloc[0]['width']
                y_center = (max(y) + min(y)) / 2 / sel.iloc[0]['height']
                h = (max(y) - min(y)) / sel.iloc[0]['height']

                # Case for missing labels

                try:
                    label = reg['region_attributes']['class name'].lower()
                except KeyError:
                    label = None
                
                if x_center is None:
                    print('help')

                bbox_info = pd.Series(name=next_id, data={
                    'image_id': sel.iloc[0].name,
                    'x': x_center,
                    'y': y_center,
                    'width': w,
                    'height': h,
                    'label': label
                })
                next_id += 1

                annotations = annotations.append(bbox_info)

    # ## Cleanup labels
    # Remove trailing and leading whitespaces, type every label with lowercase

    annotations['label'] = annotations['label'].apply(
        lambda x: x.strip().lower() if isinstance(x, str) else pd.NA)
    annotations

    # ## Remove duplicate annotations if any

    annotations = annotations.drop_duplicates()

    # # Dump annotations and images

    images.reset_index().to_json(DATA_DIR / 'images2.json.gz',
                                 orient='records',
                                 compression='gzip',
                                 default_handler=str)

    annotations.reset_index().to_json(DATA_DIR / 'annotations2.json.gz',
                                      orient='records',
                                      compression='gzip',
                                      default_handler=str)


if __name__ == '__main__':
    run()
