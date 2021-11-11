import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import yaml

def parse_args():

    argparser = argparse.ArgumentParser()

    path_args = argparser.add_argument_group('Paths')
    classes_args = argparser.add_argument_group('Selected classes')
    cfg_args = argparser.add_argument_group('Config file')

    path_args.add_argument('--root_dir',
                           action='store',
                           default='data/',
                           help='Other paths, including those are in JSON files, are relative to this.',
                           required=True)
    path_args.add_argument('--annotations',
                           action='store',
                           default='annotations.json.gz',
                           help='Path to annotations JSON file',
                           required=True)
    path_args.add_argument('--images',
                           action='store',
                           default='images.json.gz',
                           help='Path to images JSON file',
                           required=True)
    path_args.add_argument('--output_dir',
                           action='store',
                           default='yolo_dataset',
                           help='Path to the output directory',
                           required=True)
    classes_args.add_argument('-l',
                              '--selected_classes',
                              action='store',
                              nargs='+',
                              metavar=('CLASS 1', 'CLASS 2'),
                              help='Selected list of classes to extract')
    classes_args.add_argument('-u',
                              '--unlabeled',
                              action='store_true',
                              help='Include unlabeled bounding boxes as "other" superclass')
    classes_args.add_argument('-o',
                              '--other',
                              action='store_true',
                              help='Include ignored classes as "other" superclass')
    classes_args.add_argument('-e',
                              '--empty',
                              action='store_true',
                              help='Include empty images')
    cfg_args.add_argument('-c',
                          '--cfg',
                          action='store',
                          metavar='CONFIG_FILE')

    return argparser.parse_args()
    

# ----------------------------------------------------------------------------
# LEGACY CODE
# KEEP JUST IN CASE

# # Workdir. All other paths are relative to this.
# ROOT_DIR = '/raid/ruslan_bazhenov/projects/xray/cargoxray/data'

# ANNOTATIONS_FILE = 'annotations_v4.json.gz'
# IMAGES_FILE = 'images_v4.json.gz'
# YOLO_DATASET_OUTPUT_DIR = 'yolo_dataset_top6_unknowns'

# # Either a list of labels
# SELECTED_LABELS = ['shoes', 'textile',
#                    'spare parts', 'clothes', 'fabrics', 'toys']

# # Or a dictionary of labels with their future names
# SELECTED_LABELS = {'shoes': 'shoes',
#                    'textile': 'textile',
#                    'spare parts': 'spare parts',
#                    'auto parts': 'spare parts',
#                    'tools': 'spare parts',
#                    'clothes': 'clothes',
#                    'fabrics': 'fabrics',
#                    'toys': 'toys'}

# # ----------------------------------------------------------------------------

# root_dir = Path(ROOT_DIR)

# annotations_file = Path(ANNOTATIONS_FILE)
# if not annotations_file.is_absolute():
#     annotations_file = root_dir.joinpath(annotations_file)

# images_file = Path(IMAGES_FILE)
# if not images_file.is_absolute():
#     images_file = root_dir.joinpath(images_file)

# output_dir = Path(YOLO_DATASET_OUTPUT_DIR)
# if not output_dir.is_absolute():
#     output_dir = root_dir.joinpath(output_dir)

# # ----------------------------------------------------------------------------


# images = pd.read_json(images_file,
#                       orient='records',
#                       typ='frame',
#                       compression='gzip')
# images = images.set_index('image_id')

# annotations = pd.read_json(annotations_file,
#                            orient='records',
#                            typ='frame',
#                            compression='gzip')
# annotations = annotations.set_index('bbox_id')

# # ----------------------------------------------------------------------------

# if len(SELECTED_LABELS) > 0:

#     selected_labels = pd.Series(SELECTED_LABELS, name='label')

#     ignored_labels = pd.concat([selected_labels,
#                                 annotations['label'].drop_duplicates()])
#     ignored_labels = ignored_labels.drop_duplicates(keep=False)
#     ignored_labels.index = ignored_labels

# else:
#     ignored_labels = annotations[['label', 'image_id']] \
#         .drop_duplicates() \
#         .groupby('label')['image_id'] \
#         .count() \
#         .drop('unknown') \
#         .sort_values(ascending=False)[20:] \
#         .append(pd.Series([1], index=['unknown']))
#     ignored_labels.name = 'label'


# ignored_annotations = annotations \
#     .reset_index() \
#     .merge(pd.Series(ignored_labels.index, name='label')) \
#     .set_index('bbox_id')


# selected_annotations = pd.concat([annotations, ignored_annotations]) \
#     .drop_duplicates(keep=False)


# # Includes empty images

# # selected_annotations = pd.concat([selected_annotations,
# #                                   annotations[annotations['height'] == 'unknown']])

# # Includes unknown labels

# unknown_annotations = pd.concat([
#     annotations,
#     annotations[annotations['height'] == 'unknown'],
#     selected_annotations]).drop_duplicates(keep=False)

# unknown_annotations['label'] = 'unknown'

# selected_annotations = pd.concat([selected_annotations,
#                                   unknown_annotations])

# combined = selected_annotations \
#     .reset_index() \
#     .merge(images['filepath'], on='image_id') \
#     .drop_duplicates('bbox_id') \
#     .set_index('bbox_id')

# # ----------------------------------------------------------------------------

# train_images = combined.image_id.drop_duplicates().sample(frac=0.85)

# train_annotations = combined.merge(train_images, on='image_id')
# val_annotations = pd.concat(
#     [combined, train_annotations]).drop_duplicates(keep=False)

# train_images = train_annotations.filepath.drop_duplicates()
# val_images = val_annotations.filepath.drop_duplicates()

# assert len(train_annotations.image_id.drop_duplicates()) \
#     + len(
#         val_annotations.image_id.drop_duplicates()) == len(pd.concat([
#             train_annotations.image_id.drop_duplicates(),
#             val_annotations.image_id.drop_duplicates()])
#     .drop_duplicates(keep=False))

# # ----------------------------------------------------------------------------

# labels = train_annotations[train_annotations.height != 'unknown']\
#     .label\
#     .drop_duplicates()\
#     .sort_values()\
#     .reset_index(drop=True)

# output_dir.joinpath('train/images').mkdir(parents=True, exist_ok=True)
# output_dir.joinpath('train/labels').mkdir(parents=True, exist_ok=True)
# output_dir.joinpath('val/images').mkdir(parents=True, exist_ok=True)
# output_dir.joinpath('val/labels').mkdir(parents=True, exist_ok=True)

# output_dir.joinpath('cargoxray.yaml').write_text(
#     f"path: {output_dir.absolute().as_posix()}" + '\n'
#     'train: train/images' + '\n'
#     'val: val/images' + '\n'
#     f"nc: {len(labels)}" + '\n'
#     f"names: [{', '.join([f'{x}' for x in labels])}]" + '\n'
# )

# for img in tqdm(train_images, total=len(train_images)):

#     img_id = train_annotations.loc[train_annotations.filepath ==
#                                    img].iloc[0].image_id

#     os.link(root_dir.joinpath(img),
#             output_dir.joinpath(f'train/images/{img_id}{Path(img).suffix}'))

#     with output_dir.joinpath(f'train/labels/{img_id}.txt').open('w') as f:
#         for idx, row in train_annotations.loc[train_annotations.filepath == img].iterrows():
#             if row.height == 'unknown':
#                 f.write('')
#             else:
#                 f.write(
#                     f"{labels.loc[labels == row.label].index[0]} {row.x} {row.y} {row.width} {row.height}\n")


# for img in tqdm(val_images, total=len(val_images)):

#     img_id = val_annotations.loc[val_annotations.filepath ==
#                                  img].iloc[0].image_id

#     os.link(root_dir.joinpath(img),
#             output_dir.joinpath(f'val/images/{img_id}{Path(img).suffix}'))

#     with output_dir.joinpath(f'val/labels/{img_id}.txt').open('w') as f:
#         for idx, row in val_annotations.loc[val_annotations.filepath == img].iterrows():
#             if row.height == 'unknown':
#                 f.write('')
#             else:
#                 f.write(
#                     f"{labels.loc[labels == row.label].index[0]} {row.x} {row.y} {row.width} {row.height}\n")


def run(root_dir,
        annotations,
        images,
        output_dir,
        selected_classes,
        empty,
        unlabeled,
        other):

    # Load pandas dataframes
    img_frm = pd.read_json(images,
                          orient='records',
                          typ='frame',
                          compression='gzip')
    img_frm = img_frm.set_index('image_id')

    ann_frm = pd.read_json(annotations,
                               orient='records',
                               typ='frame',
                               compression='gzip')
    ann_frm = ann_frm.set_index('bbox_id')

    # Use images with annotations only
    img_frm = img_frm.loc[img_frm.index.isin(ann_frm['image_id'])]

    # Select images

    selected_images = img_frm.index[
        ~img_frm.index.isin(
            ann_frm[
                ~ann_frm['label'].isin(selected_classes)
            ]['image_id'])
    ].drop_duplicates()

    empty_images = img_frm.index[
        img_frm.index.isin(
            ann_frm[
                ann_frm['height'] == 'unknown'
            ]['image_id'])
    ].drop_duplicates()

    unlabeled_images = img_frm.index[
        img_frm.index.isin(
            ann_frm[
                (ann_frm['height'] != 'unknown')
                & (ann_frm['label'] == 'unknown')
            ]['image_id'])
    ].drop_duplicates()

    other_images = img_frm.index[
        ~img_frm.index.isin(selected_images)
        & ~img_frm.index.isin(empty_images)
        & ~img_frm.index.isin(unlabeled_images)
    ]

    if empty:
        selected_images = selected_images.append(empty_images)

    if unlabeled:
        selected_images = selected_images.append(unlabeled_images)

    if other:
        selected_images = selected_images.append(other_images)

    selected_images = selected_images.drop_duplicates()
    selected_images = img_frm.loc[selected_images].drop_duplicates('md5')

    # ------------------------------------------------------------------------
    train_images = selected_images.sample(frac=0.85)
    val_images = selected_images[~selected_images.index.isin(
        train_images.index)]

    # ----------------------------------------------------------------------------

    labels = ann_frm.loc[(ann_frm['image_id'].isin(selected_images.index))
                             & (ann_frm['height'] != 'unknown')
                             ]['label'] \
        .drop_duplicates() \
        .sort_values() \
        .reset_index(drop=True)

    breakpoint()

    output_dir.joinpath('train/images').mkdir(parents=True, exist_ok=True)
    output_dir.joinpath('train/labels').mkdir(parents=True, exist_ok=True)
    output_dir.joinpath('val/images').mkdir(parents=True, exist_ok=True)
    output_dir.joinpath('val/labels').mkdir(parents=True, exist_ok=True)

    output_dir.joinpath('cargoxray.yaml').write_text(
        f"path: {output_dir.absolute().as_posix()}" + '\n'
        'train: train/images' + '\n'
        'val: val/images' + '\n'
        f"nc: {len(labels)}" + '\n'
        f"names: [{', '.join([f'{x}' for x in labels])}]" + '\n'
    )

    for img_id in tqdm(train_images.index, total=len(train_images)):
        img_path = root_dir / img_frm.loc[[img_id]].iloc[0]['filepath']

        os.link(root_dir.joinpath(img_path),
                output_dir.joinpath(f'train/images/{img_id}{img_path.suffix}'))

        with output_dir.joinpath(f'train/labels/{img_id}.txt').open('w') as f:
            for idx, row in ann_frm.loc[ann_frm['image_id'] == img_id].iterrows():
                if row.height == 'unknown':
                    f.write('')
                else:
                    f.write(
                        f"{labels.loc[labels == row.label].index[0]} {row.x} {row.y} {row.width} {row.height}\n")

    for img_id in tqdm(val_images.index, total=len(val_images)):

        img_path = root_dir / img_frm.loc[[img_id]].iloc[0]['filepath']

        os.link(root_dir.joinpath(img_path),
                output_dir.joinpath(f'val/images/{img_id}{img_path.suffix}'))

        with output_dir.joinpath(f'val/labels/{img_id}.txt').open('w') as f:
            for idx, row in ann_frm.loc[ann_frm['image_id'] == img_id].iterrows():
                if row.height == 'unknown':
                    f.write('')
                else:
                    f.write(
                        f"{labels.loc[labels == row.label].index[0]} {row.x} {row.y} {row.width} {row.height}\n")


if __name__ == '__main__':

    args = parse_args()

    if args.cfg is not None:
        with Path(args.cfg).open('r') as f:
            cfg = yaml.load(f, yaml.Loader)
    
    else:
        cfg = vars(args)

    # Set paths to input and output files and dirs
    cfg['root_dir'] = Path(cfg['root_dir'])

    cfg['annotations'] = Path(cfg['annotations'])
    if not cfg['annotations'].is_absolute():
        cfg['annotations'] = cfg['root_dir'] / cfg['annotations']

    cfg['images'] = Path(cfg['images'])
    if not cfg['images'].is_absolute():
        cfg['images'] = cfg['root_dir'] / cfg['images']

    cfg['output_dir'] = Path(cfg['output_dir'])
    if not cfg['output_dir'].is_absolute():
        cfg['output_dir'] = cfg['root_dir'] / cfg['output_dir']

    run(**cfg)
