import json
import logging
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Union, Optional

import pandas as pd
import tqdm
from PIL import Image, UnidentifiedImageError

DATA_DIR = '/raid/ruslan_bazhenov/projects/xray/cargoxray/data'


def count_files(folder: Union[str, Path]) -> Dict[str, int]:
    """Count the number of files by extensions"""

    logging.info(f'Counting files in {folder}')
    fldr = Path(folder)
    sufcnt = {}
    for f in fldr.glob('**/*'):
        if f.is_file():
            try:
                sufcnt[f.suffix] += 1
            except KeyError:
                sufcnt[f.suffix] = 1

    return sufcnt


def load_frame(path_to_frame: Optional[Union[str, Path]] = None) \
        -> pd.DataFrame:
    """Load a dataframe if exists, or generate new"""

    logging.info(f'Loading frame {path_to_frame}')
    try:
        frame = pd.read_json(path_to_frame,
                             orient='records',
                             typ='frame',
                             compression='gzip')

    except:
        frame = pd.DataFrame()

    return frame


def scan_images(images_dir: Union[str, Path],
                data_dir: Union[str, Path],
                images_frame: pd.DataFrame) \
        -> pd.DataFrame:
    """Scan "images_dir" for .tif files, save info about the file in 
    "images_frame". Image paths are stored relative to "data_dir". """

    logging.info(f'Scanning images in {images_dir}')

    img_dir = Path(images_dir)
    img_frm = images_frame.copy()

    if len(img_frm) > 0:
        next_id = img_frm.index.max() + 1
    else:
        next_id = 0

    image_path: Path
    for image_path in tqdm.tqdm(list(img_dir.glob('**/*.tif')), desc='Images'):

        rel_path = image_path.relative_to(data_dir)
        logging.debug(rel_path)

        # Skip already imported images
        if len(img_frm) > 0:
            if len(img_frm.loc[img_frm['filepath'] == rel_path.as_posix()]) > 0:
                continue

        try:
            image = Image.open(image_path)
        except UnidentifiedImageError:
            logging.error(f"Corrupted image \"{image_path}\"")
            continue

        image_info = pd.Series(name=next_id, data={

            'md5': md5(image_path.read_bytes()).hexdigest(),
            'size': image_path.stat().st_size,
            'width': image.width,
            'height': image.height,
            'filepath': rel_path.as_posix(),

        })

        next_id += 1

        image.close()

        img_frm = img_frm.append(image_info)

    return img_frm


def scan_annotations(annotations_dir: Union[str, Path],
                     annotations_frame: pd.DataFrame,
                     images_frame: pd.DataFrame) \
        -> pd.DataFrame:
    """ Scan "annotations_dir" for .json files and append annotations to 
    "annotations_frame". Images are linked to annotations via
    "image_id" and "images_frame"."""

    logging.info(f'Scanning annotations in {annotations_dir}')
    ann_dir = Path(annotations_dir)
    ann_frm = annotations_frame.copy()
    img_frm = images_frame.copy()

    if len(ann_frm) > 0:
        next_id = img_frm.index.max() + 1
    else:
        next_id = 0

    #  Scan & import JSON files
    img_frm = img_frm.assign(
        filename=img_frm['filepath'].apply(lambda x: Path(x).name))

    for src in tqdm.tqdm(list(ann_dir.glob('**/*.json')),
                         desc='JSONs'):

        logging.debug(f'Scanning annotations in {src}')
        # Annotations loaded
        with src.open('rb') as fs:
            anns = json.load(fs)

        # For each group of annotations for one image
        ann: Dict[str, Any]
        for ref, ann in anns.items():

            # Try to find the corresponding image
            sel = img_frm.loc[(img_frm['filename'] == ann['filename'])
                              & (img_frm['size'] == ann['size'])]

            sel = sel.drop_duplicates('md5')

            # Corresponding image not found, skip the annotation
            if len(sel) == 0:
                continue
            # More than one image found, warning
            elif len(sel) > 1:
                logging.warning(sel['filepath'].to_string())

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
                if reg['shape_attributes']['name'] == 'polyline' \
                        or reg['shape_attributes']['name'] == 'polygon':
                    x_points = reg['shape_attributes']['all_points_x']
                    y_points = reg['shape_attributes']['all_points_y']

                elif reg['shape_attributes']['name'] == 'rect':
                    x_points = reg['shape_attributes']['x']
                    y_points = reg['shape_attributes']['y']
                    w = reg['shape_attributes']['width']
                    h = reg['shape_attributes']['height']

                    x_points = [x_points, x_points + w, x_points + w, x_points]
                    y_points = [y_points, y_points, y_points + h, y_points + h]
                else:
                    logging.warning(
                        f'Unexpected annotation shape in '
                        f'"{src}", "{ann["filename"]}{ann["size"]}". '
                        f'Found {reg["shape_attributes"]["name"]}')
                    continue

                x = (max(x_points) + min(x_points)) / 2 / sel.iloc[0]['width']
                w = (max(x_points) - min(x_points)) / sel.iloc[0]['width']

                y = (max(y_points) + min(y_points)) / 2 / sel.iloc[0]['height']
                h = (max(y_points) - min(y_points)) / sel.iloc[0]['height']

                # Case for missing labels

                try:
                    label = reg['region_attributes']['class name'].lower()
                except KeyError:
                    label = None

                bbox_info = pd.Series(name=next_id,
                                      data={
                                          'image_id': int(sel.iloc[0].name),
                                          'x': x,
                                          'y': y,
                                          'width': w,
                                          'height': h,
                                          'label': label
                                      })
                next_id += 1

                ann_frm = ann_frm.append(bbox_info)

    return ann_frm


def cleanup_images(images_frame: pd.DataFrame) -> pd.DataFrame:
    img_frm = images_frame.copy()

    img_frm = img_frm\
        .drop(columns=['filepath'])\
        .drop_duplicates('md5')\
        .reset_index()\
        .merge(img_frm[['md5', 'filepath']],
               on='md5') \
        .set_index('image_id') \
        .drop_duplicates()

    return img_frm


def cleanup_annotations(annotations: pd.DataFrame) -> pd.DataFrame:

    ann_frm = annotations.copy()

    # Strip and lowercase everything
    ann_frm['label'] = ann_frm['label'].apply(
        lambda x: x.strip().lower() if isinstance(x, str) else pd.NA)

    # Load manual label typos fixes dictionary
    label_mappings_fix = pd.read_csv(
        '/raid/ruslan_bazhenov/projects/xray/cargoxray/utils/'
        'label_mappings_fix.csv',
        names=['original', 'typos', 'merge']
    ).set_index('original')

    # Apply manual label fixes
    ann_frm = ann_frm.replace(label_mappings_fix['typos'].to_dict())

    # Drop duplicates
    ann_frm = ann_frm.drop_duplicates()

    return ann_frm


def save_frame(frame: pd.DataFrame, destination: Union[str, Path]) -> None:
    logging.info(f'Saving frame {destination}')
    frame.reset_index().to_json(destination,
                                orient='records',
                                compression='gzip',
                                default_handler=str)


def run(DATA_DIR):

    logging.info('Start')

    data_dir = Path(DATA_DIR)

    sufcnt = count_files(data_dir / 'images')

    for k, v in sufcnt.items():
        print(f'{k.strip("."):>10} : {v}')

    #  Import images

    #  Load existsing images databases if exist
    images = load_frame(data_dir / 'images_v2.json.gz')
    if len(images) > 0:
        images = images.set_index('image_id')
    else:
        images.index.name = 'image_id'

    #  Scan & import image files
    images = scan_images(data_dir / 'images', data_dir, images)

    #  Map duplicate images to the same location
    images = cleanup_images(images)

    #  Import annotations

    #  Load existing annotations files

    annotations = load_frame(None)
    if len(annotations) > 0:
        annotations = annotations.set_index('bbox_id')
    else:
        annotations.index.name = 'bbox_id'

    annotations = scan_annotations(data_dir / 'images',
                                   annotations,
                                   images)

    # Refine annotations labels
    annotations = cleanup_annotations(annotations)

    save_frame(images, data_dir / 'images_v3.json.gz')
    save_frame(annotations, data_dir / 'annotations_v3.json.gz')

    logging.info('Finish')


if __name__ == '__main__':
    logging.basicConfig(filename=f'{Path(__file__)}.log',
                        level=logging.INFO)
    run(DATA_DIR)
