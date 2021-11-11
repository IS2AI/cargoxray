import json
import logging
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional
import profile

import pandas as pd
import tqdm
from PIL import Image, UnidentifiedImageError

IMAGE_FORMATS = {'.tif', '.tiff', '.jpeg', '.jpg'}


def scan_images(images_dir: Union[str, Path],
                _root_dir: Union[str, Path],
                images_frame: pd.DataFrame) \
        -> pd.DataFrame:
    """Scan "images_dir" for .tif files, save info about the file in 
    "images_frame". Image paths are stored relative to "_root_dir". """

    logging.info(f'Scanning images in {images_dir}')

    img_dir = Path(images_dir)
    img_frm = images_frame.copy()

    if len(img_frm) > 0:
        next_id = img_frm.index.max() + 1
    else:
        next_id = 0

    image_path: Path
    for image_path in tqdm.tqdm(list(img_dir.glob('**/*'))[:200], desc='Images'):

        if image_path.suffix.lower() in IMAGE_FORMATS:

            rel_path = image_path.relative_to(_root_dir)
            logging.debug(rel_path)

            # Skip already imported images
            if rel_path in img_frm['filepath'].values:
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
                     root_dir: Union[str, Path],
                     images_frame: pd.DataFrame,
                     annotations_frame: pd.DataFrame,
                     json_files_frame: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan "annotations_dir" for .json files and append annotations to 
    "annotations_frame". Images are linked to annotations via
    "image_id" and "images_frame".
    """

    logging.info(f'Scanning annotations in {annotations_dir}')

    _annotations_dir = Path(annotations_dir)
    _root_dir = Path(root_dir)

    _annotations_frame = annotations_frame.copy()
    _images_frame = images_frame.copy()
    _json_files_frame = json_files_frame.copy()

    if len(_annotations_frame) > 0:
        annotations_next_id = _annotations_frame.index.max() + 1
    else:
        annotations_next_id = 0

    if len(_json_files_frame) > 0:
        json_files_next_id = _json_files_frame.index.max() + 1
    else:
        json_files_next_id = 0

    # Scan & import JSON files
    _images_frame = _images_frame.assign(
        filename=_images_frame['filepath'].apply(lambda x: Path(x).name))

    src: Path
    for src in tqdm.tqdm(list(_annotations_dir.glob('**/*.json'))[:10],
                         desc='JSONs'):

        logging.debug(f'Scanning annotations in {src}')

        rel_path = src.relative_to(_root_dir)
        raw_bytes = src.read_bytes()
        md5_hash = md5(raw_bytes).hexdigest()

        # Skip already imported JSON files
        if md5_hash in _json_files_frame['md5'].values:
            continue

        # Annotations loaded
        anns = json.loads(raw_bytes)

        # For each group of annotations for one image
        ann: Dict[str, Any]
        for ref, ann in anns.items():

            # Try to find the corresponding image
            sel = _images_frame.loc[(_images_frame['filename'] == ann['filename'])
                                    & (_images_frame['size'] == ann['size'])]

            sel = sel.drop_duplicates('md5')

            # Corresponding image not found, skip the annotation
            if len(sel) == 0:
                logging.warning(f'Missing image {ref}')
                continue
            # More than one image found, warning
            elif len(sel) > 1:
                logging.error(f'More than image corresponding to {ref}')

            # In some JSON files regions are dict, in some are lists
            if isinstance(ann['regions'], dict):
                regions = list(ann['regions'].values())
            else:
                regions = ann['regions']

            # For each bbox
            for reg in regions:

                # Skip missing bboxes
                if reg is None:
                    logging.warn(f'Skipping broken bbox in {rel_path}, {ref}')
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
                        f'Unexpected annotation shape in {rel_path}, {ref}. '
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
                    label = pd.NA

                bbox_info = pd.Series(name=annotations_next_id,
                                      data={
                                          'image_id': int(sel.iloc[0].name),
                                          'x': x,
                                          'y': y,
                                          'width': w,
                                          'height': h,
                                          'label': label
                                      })
                annotations_next_id += 1

                _annotations_frame = _annotations_frame.append(bbox_info)

        json_info = pd.Series(name=json_files_next_id,
                              data={
                                  'filepath': rel_path.as_posix(),
                                  'md5': md5_hash
                              })
        json_files_next_id += 1

        _json_files_frame = _json_files_frame.append(json_info)

    return (_annotations_frame, _json_files_frame,)


def cleanup_images(images_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate images with identical md5 into one image_id"""

    img_frm = images_frame.copy()

    img_frm = img_frm \
        .drop(columns=['filepath']) \
        .drop_duplicates('md5') \
        .reset_index() \
        .merge(img_frm[['md5', 'filepath']],
               on='md5') \
        .set_index('image_id') \
        .drop_duplicates()

    return img_frm


def cleanup_annotations(annotations: pd.DataFrame,
                        label_fixes_path: Union[Path, str]) -> pd.DataFrame:

    _annotations = annotations.copy()

    # Strip and lowercase everything
    _annotations['label'] = _annotations['label'].apply(
        lambda x: x.strip().lower() if isinstance(x, str) else pd.NA)

    # Load manual label typos fixes dictionary
    fixes = pd.read_csv(label_fixes_path,
                        names=['original', 'typos', 'merge']
                        ).set_index('original')

    # Apply manual label fixes
    _annotations = _annotations.replace(fixes['typos'].to_dict())

    # Drop duplicates
    _annotations = _annotations.drop_duplicates()

    return _annotations


def add_empty(annotations: pd.DataFrame,
              images: pd.DataFrame) -> pd.DataFrame:

    ann_frm = annotations.copy()
    img_frm = images.copy()

    if len(ann_frm) > 0:
        next_id = img_frm.index.max() + 1
    else:
        next_id = 0

    for idx, img in img_frm[img_frm['filepath']
                            .apply(lambda x: x.find('empty/') >= 0)]\
            .iterrows():
        ann = pd.Series(
            name=next_id,
            data={'image_id': img.name})
        next_id += 1
        ann_frm = ann_frm.append(ann)

    return ann_frm


def run(root_dir: Union[Path, str],
        images_dir: Union[Path, str],
        images_frame_path: Union[Path, str],
        annotations_frame_path: Union[Path, str],
        json_files_frame_path: Union[Path, str],
        label_fixes_path: Union[Path, str]) -> Dict[str, int]:
    """
        Scan the images_dir for images and JSON files. All paths 
        except labels_fixes are absolute or relative to root_dir. 

    """

    logging.info('Start')

    # Convert str to Path and make relative paths absolute

    _root_dir = Path(root_dir)

    _images_dir = Path(images_dir)
    if not _images_dir.is_absolute():
        _images_dir = _root_dir / _images_dir

    _images_frame_path = Path(images_frame_path)
    if not _images_frame_path.is_absolute():
        _images_frame_path = _root_dir / _images_frame_path

    _annotations_frame_path = Path(annotations_frame_path)
    if not _annotations_frame_path.is_absolute():
        _annotations_frame_path = _root_dir / _annotations_frame_path

    _json_files_frame_path = Path(json_files_frame_path)
    if not _json_files_frame_path.is_absolute():
        _json_files_frame_path = _root_dir / _json_files_frame_path

    _label_fixes_path = Path(label_fixes_path)
    if not _label_fixes_path.is_absolute():
        _label_fixes_path = _root_dir / _label_fixes_path

    # Import images

    # Load existsing images databases if exist

    logging.info(f'Loading frame {_images_frame_path}')
    try:
        images = pd.read_json(_images_frame_path,
                              orient='records',
                              typ='frame',
                              compression='gzip')
    except:
        images = pd.DataFrame(
            columns=['image_id', 'md5', 'size', 'width', 'height', 'filepath'])

    images = images.set_index('image_id')

    # Scan & import image files
    images = scan_images(_images_dir,
                         _root_dir,
                         images)

    # Map duplicate images to the same image_id
    images = cleanup_images(images)

    # Import annotations

    # Load existing annotations and information on processed json files

    logging.info(f'Loading frame {_annotations_frame_path}')
    try:
        annotations = pd.read_json(_annotations_frame_path,
                                   orient='records',
                                   typ='frame',
                                   compression='gzip')

    except:
        annotations = pd.DataFrame(
            columns=['bbox_id', 'image_id', 'x', 'y', 'width', 'height', 'label'])

    annotations = annotations.set_index('bbox_id')

    logging.info(f'Loading frame {_json_files_frame_path}')
    try:
        json_files = pd.read_json(_json_files_frame_path,
                                  orient='records',
                                  typ='frame',
                                  compression='gzip')

    except:
        json_files = pd.DataFrame(columns=['json_id', 'filepath', 'md5'])
    
    json_files = json_files.set_index('json_id')

    annotations = add_empty(annotations, images)

    annotations, json_files = scan_annotations(_images_dir,
                                               _root_dir,
                                               images,
                                               annotations,
                                               json_files)
    # Refine annotations labels
    annotations = cleanup_annotations(annotations, label_fixes_path)

    logging.info(f'Saving frame {_images_frame_path}')
    images.reset_index().to_json(_images_frame_path,
                                 orient='records',
                                 compression='gzip',
                                 default_handler=str)

    logging.info(f'Saving frame {_annotations_frame_path}')
    annotations.reset_index().to_json(_annotations_frame_path,
                                      orient='records',
                                      compression='gzip',
                                      default_handler=str)

    logging.info(f'Saving frame {_json_files_frame_path}')
    json_files.reset_index().to_json(_json_files_frame_path,
                                     orient='records',
                                     compression='gzip',
                                     default_handler=str)

    logging.info('Finish')


if __name__ == '__main__':
    logging.basicConfig(filename=f'{Path(__file__)}.log',
                        level=logging.INFO)

    cfg = dict(
        root_dir='/raid/ruslan_bazhenov/projects/xray/cargoxray/data/',
        images_dir='images/',
        images_frame_path='images.json.gz',
        annotations_frame_path='annotations.json.gz',
        json_files_frame_path='json_files.json.gz',
        label_fixes_path='/raid/ruslan_bazhenov/projects/xray/cargoxray/utils/label_mappings_fix.csv'
    )

    run(**cfg)
