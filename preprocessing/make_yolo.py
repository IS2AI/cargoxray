import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_args():

    argparser = argparse.ArgumentParser()

    path_args = argparser.add_argument_group('Paths')

    path_args.add_argument('--root_dir',
                           action='store',
                           default='data/',
                           help='Other paths, including those are in JSON files, are relative to this.',
                           required=False)

    path_args.add_argument('--annotations_file',
                           action='store',
                           default='annotations.json.gz',
                           help='Path to annotations JSON file',
                           required=False)

    path_args.add_argument('--images_file',
                           action='store',
                           default='images.json.gz',
                           help='Path to images JSON file',
                           required=False)

    path_args.add_argument('--output_dir',
                           action='store',
                           default='yolo_dataset',
                           help='Path to the output directory',
                           required=True)

    mode_args = argparser.add_argument_group('Mode')
    
    mode_args.add_argument('--hardlink',
                           action='store_true',
                           help='Copy files by generating hardlinks',
                           required=False)
     
    mode_args.add_argument('--symlink',
                           action='store_true',
                           help='Copy files by generating symbolic links',
                           required=False)
    
    mode_args.add_argument('--copy',
                           action='store_true',
                           help='Copy files by simple copy. Requires more disk spaÎce!',
                           required=False)

    classes_args = argparser.add_argument_group('Selected classes')

    classes_args.add_argument('--selected_classes',
                              action='store',
                              nargs='+',
                              metavar=('CLASS 1', 'CLASS 2'),
                              help='Selected list of classes to extract')
    classes_args.add_argument('--unlabeled',
                              action='store_true',
                              help='Include unlabeled bounding boxes as "other" superclass')
    classes_args.add_argument('--other',
                              action='store_true',
                              help='Include ignored classes as "other" superclass')
    classes_args.add_argument('--empty',
                              action='store_true',
                              help='Include empty images')

    return argparser.parse_args()


def run(root_dir,
        annotations_file,
        images_file,
        output_dir,
        mode,
        selected_classes,
        include_empty,
        include_unlabeled,
        include_other):

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
