import hashlib
import json
import logging
from os import R_OK
import pathlib
import shutil
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
from numpy import empty, select

import pandas as pd
from PIL import Image, UnidentifiedImageError
from pandas.core.indexes.range import RangeIndex
from tqdm import tqdm

import config
import random
import utils


class Cargoxray:

    _img_dir: Path
    _images_json_path: Path
    _annotations_json_path: Path
    _categories_json_path: Path

    _images: pd.DataFrame
    _annotations: pd.DataFrame
    _categories: pd.DataFrame

    _images_next_id: int
    _annotations_next_id: int
    _categories_next_id: int

    _cache_ref: Dict[str, Path]
    _cache_md5: Dict[Path, str]
    _cache_copy: List[Tuple[Path, Path]]

    _label_replacements: Dict[str, str]

    def __init__(self,
                 img_dir: Union[Path, str],
                 images_json_path: Union[Path, str],
                 annotations_json_path: Union[Path, str],
                 categories_json_path: Union[Path, str]):

        self._img_dir = Path(img_dir)
        self._images_json_path = Path(images_json_path)
        self._annotations_json_path = Path(annotations_json_path)
        self._categories_json_path = Path(categories_json_path)

        self._cache_copy = []
        self._cache_md5 = {}
        self._cache_ref = {}

        self._label_replacements = utils.load_label_replacements(
            'preprocessing/label_mappings_fix.csv')

        self._images = utils.load_or_create_frame(
            path=self._images_json_path,
            columns=config.IMAGES_FRAME_COLUMNS,
            index=config.IMAGES_FRAME_INDEX)

        self._annotations = utils.load_or_create_frame(
            path=self._annotations_json_path,
            columns=config.ANNOTATIONS_FRAME_COLUMNS,
            index=config.ANNOTATIONS_FRAME_INDEX)

        self._categories = utils.load_or_create_frame(
            path=self._categories_json_path,
            columns=config.CATEGORIES_FRAME_COLUMNS,
            index=config.CATEGORIES_FRAME_INDEX)

        self._images_next_id = self._images.index.max() + 1\
            if len(self._images) > 0 else 0

        self._annotations_next_id = self._annotations.index.max() + 1\
            if len(self._annotations) > 0 else 0

        self._categories_next_id = self._categories.index.max() + 1\
            if len(self._categories) > 0 else 0

    def import_data(self,
                    import_dir: Optional[Union[Path, str]] = None,
                    empty_images_dir: Optional[Union[Path, str]] = None)\
            -> Dict:
        """Import a folder into the dataset. `import_dir` is for annotated
        images and `empty_images_dir` is for null (empty, background only)
        images. At least one of two arguments must be provided.
        Returns a statistics on the imported images and annotations.
        """

        if import_dir is None and empty_images_dir is None:
            raise ValueError('One of the arguments must be non None')

        if import_dir is not None:
            import_dir = Path(import_dir)

            self._cache_ref = utils.make_ref_cache(import_dir)

            for json_file in tqdm(list(import_dir.glob('**/*.json')),
                                  desc='All JSON files'):
                try:
                    self._import_json_file(json_file)
                except Exception as e:
                    logging.error(f'Failed to process {json_file}. {e}')

        if empty_images_dir is not None:
            empty_images_dir = Path(empty_images_dir)

            try:
                self._import_empty_images(empty_images_dir)
            except Exception as e:
                logging.error(f'Could import empty images. {e}')

        self._annotations = self._annotations.drop_duplicates()
        self._images = self._images.drop_duplicates()

    def apply_changes(self):

        if not self._img_dir.exists():
            self._img_dir.mkdir(parents=True)

        for src, dst in tqdm(self._cache_copy,
                             desc='Copying new images'):
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(e)

        if self._images_json_path.exists():
            self._images_json_path.rename(
                self._images_json_path.as_posix() + '.bak')

        if self._annotations_json_path.exists():
            self._annotations_json_path.rename(
                self._annotations_json_path.as_posix() + '.bak')

        if self._categories_json_path.exists():
            self._categories_json_path.rename(
                self._categories_json_path.as_posix() + '.bak')

        self._images.reset_index().to_json(self._images_json_path,
                                           orient='records',
                                           compression='gzip')

        self._annotations.reset_index().to_json(self._annotations_json_path,
                                                orient='records',
                                                compression='gzip')

        self._categories.reset_index().to_json(self._categories_json_path,
                                               orient='records',
                                               compression='gzip')

    def _import_json_file(self, json_file: Path):
        """Parses JSON file and imports new annotations. New images linked to
        the annotations are automatically imported and copied
        to the data directory of the dataset.
        """

        with json_file.open('r') as fp:
            data = json.load(fp)

        for img_ref, img_anns in tqdm(data.items(),
                                      desc='Current JSON file'):

            img_path = self._get_image_path_by_ref(img_ref)
            if img_path is None:
                logging.error(f'Could not find image {img_ref}')
                continue

            # In some JSON files regions are dict, in some are lists
            if isinstance(img_anns['regions'], dict):
                regions = list(img_anns['regions'].values())
            else:
                regions = img_anns['regions']

            for region in regions:
                try:
                    bbox, label = utils.parse_region(region)
                    self._add_annotation(img_path, bbox, label)
                except Exception as e:
                    logging.error(f'Error during parsing "{json_file}", '
                                  f'"{img_ref}". {e}')

    def _add_annotation(self,
                        img_path: Path,
                        bbox: List[int],
                        label: str) -> int:

        image_id = self._add_image(img_path)

        category_id = self._add_category(label)

        new_annotation = self._append_new_annotation(image_id,
                                                     bbox,
                                                     category_id)

        annotation_id = new_annotation.name
        is_new = True

        return annotation_id

    def _append_new_annotation(self,
                               image_id: int,
                               bbox: List[int],
                               category_id: int) -> pd.Series:

        new_annotation = pd.Series(
            name=self._annotations_next_id,
            data={
                'image_id': image_id,
                'x': bbox[0],
                'y': bbox[1],
                'w': bbox[2],
                'h': bbox[3],
                'category_id': category_id
            })
        self._annotations_next_id += 1

        self._annotations = self._annotations.append(new_annotation)

        return new_annotation

    def _get_image_path_by_ref(self, img_ref: str) -> Optional[Path]:

        try:
            img_path = self._cache_ref[img_ref]
        except KeyError:
            img_path = None

        return img_path

    def _get_image_id_by_path(self, img_path: Union[Path, str])\
            -> Optional[int]:

        img_md5 = self._get_md5(img_path)

        sel = self._images.loc[self._images['md5'] == img_md5]

        if len(sel) > 0:
            image_id = sel.iloc[0].name
        else:
            image_id = None

        return image_id

    def _add_image(self, img_path) -> int:

        image_id = self._get_image_id_by_path(img_path)
        is_new = False

        if image_id is None:
            new_image = self._append_new_image(img_path)
            image_id = new_image.name
            is_new = True

        return image_id

    def _append_new_image(self, img_path: Path) -> pd.Series:

        image_id = self._images_next_id
        self._images_next_id += 1

        file_name = f'{image_id:05X}{img_path.suffix}'

        image = Image.open(img_path)
        image.verify()

        new_image = pd.Series(
            name=image_id,
            data={
                'file_name': file_name,
                'height': image.height,
                'width': image.width,
                'md5': self._get_md5(img_path),
                'size': img_path.stat().st_size,
            }
        )

        self._cache_copy.append((img_path,
                                self._img_dir / file_name))

        self._images = self._images.append(new_image)

        return new_image

    def _get_md5(self, file_path: Union[Path, str]) -> str:
        file_path = Path(file_path).absolute()

        try:
            file_md5 = self._cache_md5[file_path.as_posix()]
        except KeyError:
            file_md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
            self._cache_md5[file_path.as_posix()] = file_md5

        return file_md5

    def _get_category_id_by_label(self, label: str) -> int:

        if label is None:
            sel = self._categories.loc[self._categories['name'].isna()]
        else:
            sel = self._categories.loc[self._categories['name'] == label]

        if len(sel) > 0:
            category_id = sel.iloc[0].name
        else:
            category_id = None

        return category_id

    def _add_category(self, label: str) -> int:

        label = utils.fix_label(label,
                                self._label_replacements)

        category_id = self._get_category_id_by_label(label)
        is_new = False

        if category_id is None:
            new_category = self._append_new_category(label)
            category_id = new_category.name
            is_new = True

        return category_id

    def _rename_categories(self, mapping: Dict[str, str]):

        self._categories['name'] = self._categories['name'].replace(mapping)

        z = self._categories.drop_duplicates('name')\
            .reset_index(drop=True)\
            .reset_index()\
            .merge(
                self._categories
                .reset_index(),
                on='name',
                how='inner')\
            .rename(columns={
                self._categories.index.name: 'old_index',
                'index': 'category_id'
            })

        self._annotations['category_id'] =\
            self._annotations['category_id'].replace(
                z.set_index('old_index')['category_id'].to_dict())

        self._categories = z\
            .set_index(self._categories.index.name)\
            .drop(columns='old_index')\
            .drop_duplicates()\
            .sort_index()

    def _append_new_category(self, label: str) -> pd.Series:

        new_category = pd.Series(
            name=self._categories_next_id,
            data={
                'name': label
            }
        )
        self._categories_next_id += 1

        self._categories = self._categories.append(new_category)
        return new_category

    def _import_empty_images(self, empty_images_dir: Path):

        for img_path in empty_images_dir.glob('**/*'):

            if img_path.is_file()\
                    and img_path.suffix in config.IMAGE_FORMATS:
                self._add_image(img_path)

    def export_data(self,
                    export_dir: Union[Path, str],
                    selected_labels: Union[List[str], Dict[str, str]],
                    include_empty: bool,
                    splits_names: List[str],
                    splits_frac: List[float],
                    copy_func):

        export_dir = Path(export_dir)

        assert len(splits_names) == len(splits_frac)
        assert sum(splits_frac) == 1

        # Same structure as self._categories
        # but has additional field "yolo_id"
        # ["id" (index), "name", "yolo_id"]
        # "id" is an original category id
        # "name" is the new name according to selected_labels
        # "yolo_id" is new id, generated after renaming with selected_labels

        categories =\
            self._select_categories(selected_labels)

        image_ids = self._select_images(
            categories,
            include_empty)

        image_splits = utils.split(
            image_ids,
            splits_frac)

        for split, sname in zip(image_splits, splits_names):

            self._export_yolo_images(
                split,
                export_dir / sname,
                copy_func,
                categories)

        self._export_yolo_config(export_dir,
                                 splits_names,
                                 categories)

    def _select_categories(
        self,
        labels: Union[List[str], Dict[str, str]])\
            -> pd.DataFrame:

        if not isinstance(labels, dict):
            labels = {x: x for x in labels}

        # Select categories by dictionary keys
        categories = self._categories.loc[
            self._categories['name'].isin(labels.keys())
        ]

        # Rename categories names according to dictionary
        categories['name'] = categories['name'].map(labels)

        # Generate YOLO id
        categories = categories.reset_index().merge(
            right=categories[['name']]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={'index': 'yolo_id'}),
            on='name',
            how='inner')\
            .set_index('category_id')

        return categories

    def _select_images(self,
                       categories: pd.DataFrame,
                       include_empty: bool = False) -> List[int]:

        category_ids = categories.index.tolist()

        ignored_annotations = self._annotations.loc[
            ~self._annotations['category_id'].isin(category_ids)]

        ignored_image_ids = ignored_annotations['image_id'].drop_duplicates()

        selected_annotations = self._annotations.loc[
            ~self._annotations['image_id'].isin(ignored_image_ids)]

        assert selected_annotations['category_id'].isin(category_ids).all()

        selected_images = selected_annotations['image_id']

        if include_empty:
            empty_images = self._images.loc[
                ~self._images.index.isin(self._annotations['image_id'])]

            selected_images = selected_images.append(
                pd.Series(empty_images.index))

        return selected_images.drop_duplicates().to_list()

    def _export_yolo_config(self,
                            path: Path,
                            splits_names: List[str],
                            categories: pd.DataFrame):

        if not path.exists():
            path.mkdir(parents=True)

        categories = categories.sort_values('yolo_id')

        config_path = path.joinpath('dataset.yaml')

        # path: /raid/ruslan_bazhenov/projects/xray/cargoxray/data/test_yolo
        # train: train/images
        # val: val/images
        # nc: 3
        # names: [shoes, spare parts, toys]

        config = {}

        config['path'] = path.as_posix()

        for sname in splits_names:
            config[sname] = f'{sname}/images'

        config['nc'] = str(len(categories.drop_duplicates('yolo_id')))
        config['names'] = '[{}]'.format(', '.join(
            categories
            .drop_duplicates('yolo_id')
            .sort_values('yolo_id')['name'].to_list()
        )
        )

        config_path.write_text(
            '\n'.join([f'{k}: {v}'
                       for k, v in config.items()]))

    def _export_yolo_images(self,
                            image_ids: int,
                            path: Path,
                            copy_func,
                            categories: pd.DataFrame):

        path.joinpath('images').mkdir(parents=True)
        path.joinpath('labels').mkdir(parents=True)

        for image_id in image_ids:

            image_info = self._images.loc[self._images.index ==
                                          image_id].iloc[0]

            yolo_text = self._make_yolo_text(image_info,
                                             categories)

            image_path = path.joinpath(
                'images',
                image_info['file_name'])

            text_path = path.joinpath(
                'labels',
                f'{image_path.stem}.txt')

            text_path.write_text(yolo_text)
            copy_func(self._img_dir / image_path.name,
                      image_path)

    def _make_yolo_text(self,
                        image_info: pd.Series,
                        categories: pd.DataFrame) -> str:

        txt = []

        sel = self._annotations.loc[
            self._annotations['image_id'] == image_info.name]

        sel = sel.join(categories,
                       on='category_id',
                       how='inner')

        for _, row in sel.iterrows():

            x, y, w, h = utils.convert_to_yolo(
                bbox=(
                    row['x'],
                    row['y'],
                    row['w'],
                    row['h']
                ),
                image_shape=(
                    image_info['width'],
                    image_info['height']
                )
            )

            txt.append("{} {} {} {} {}".format(
                row['yolo_id'],
                x, y, w, h
            ))

        txt = '\n'.join(txt) + '\n'

        return txt
