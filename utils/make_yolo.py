import math
import shutil
from pathlib import Path

import albumentations as alb
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from PIL import Image
from torchvision import transforms as tf

DATA_DIR = '/home/ruslan_bazhenov/raid/projects/xray/cargoxray/data'

data_dir = Path(DATA_DIR)
yolo_dir = data_dir / 'yolo_dataset_v3'
yolo_dir.mkdir()


images = pd.read_json('data/images_v3.json.gz',
                      orient='records',
                      typ='frame',
                      compression='gzip')
images = images.set_index('image_id')

annotations = pd.read_json('data/annotations_v3.json.gz',
                           orient='records',
                           typ='frame',
                           compression='gzip')
annotations = annotations.set_index('bbox_id')


ignored_labels = annotations[['label', 'image_id']]\
    .drop_duplicates()\
    .groupby('label')['image_id']\
    .count()\
    .drop('unknown')\
    .sort_values(ascending=False)[20:]\
    .append(pd.Series([1], index=['unknown']))
ignored_labels.name = 'label'


ignored_annotations = annotations\
    .reset_index()\
    .merge(pd.Series(ignored_labels.index, name='label'))\
    .set_index('bbox_id')


selected_annotations = pd.concat([annotations, ignored_annotations]) \
    .drop_duplicates(keep=False)


combined = selected_annotations\
    .reset_index()\
    .merge(images['filepath'], on='image_id')\
    .drop_duplicates('bbox_id')\
    .set_index('bbox_id')

# ----------------------------------------------------------------------------

train_images = combined.image_id.drop_duplicates().sample(frac=0.85)

train_annotations = combined.merge(train_images, on='image_id')
val_annotations = pd.concat(
    [combined, train_annotations]).drop_duplicates(keep=False)

train_images = train_annotations.filepath.drop_duplicates()
val_images = val_annotations.filepath.drop_duplicates()

assert len(train_annotations.image_id.drop_duplicates()) \
    + len(
        val_annotations.image_id.drop_duplicates()) == len(pd.concat([
            train_annotations.image_id.drop_duplicates(),
            val_annotations.image_id.drop_duplicates()])
    .drop_duplicates(keep=False))

# ----------------------------------------------------------------------------

labels = train_annotations\
    .label\
    .drop_duplicates()\
    .sort_values()\
    .reset_index(drop=True)

yolo_dir.joinpath('cargoxray.yaml').write_text(
    f"path: {yolo_dir.absolute().as_posix()}" + '\n'
    'train: train/images' + '\n'
    'test: test/images' + '\n'
    f"nc: {len(labels)}" + '\n'
    f"names: [{', '.join([f'{x}' for x in labels])}]" + '\n'
)


yolo_dir.joinpath('train/images').mkdir(parents=True, exist_ok=True)
yolo_dir.joinpath('train/labels').mkdir(parents=True, exist_ok=True)
yolo_dir.joinpath('val/images').mkdir(parents=True, exist_ok=True)
yolo_dir.joinpath('val/labels').mkdir(parents=True, exist_ok=True)

for img in tqdm.tqdm(train_images, total=len(train_images)):

    img_id = train_annotations.loc[train_annotations.filepath ==
                                   img].iloc[0].image_id

    shutil.copy(data_dir.joinpath(img),
                yolo_dir.joinpath(f'train/images/{img_id}.tif'))

    with yolo_dir.joinpath(f'train/labels/{img_id}.txt').open('w') as f:
        for idx, row in train_annotations.loc[train_annotations.filepath == img].iterrows():
            f.write(
                f"{labels.loc[labels == row.label].index[0]} {row.x} {row.y} {row.width} {row.height}\n")


for img in tqdm.tqdm(val_images, total=len(val_images)):

    img_id = val_annotations.loc[val_annotations.filepath ==
                                   img].iloc[0].image_id

    shutil.copy(data_dir.joinpath(img),
                yolo_dir.joinpath(f'val/images/{img_id}.tif'))

    with yolo_dir.joinpath(f'val/labels/{img_id}.txt').open('w') as f:
        for idx, row in val_annotations.loc[val_annotations.filepath == img].iterrows():
            f.write(
                f"{labels.loc[labels == row.label].index[0]} {row.x} {row.y} {row.width} {row.height}\n")

exit()
bbox_params = alb.BboxParams(format='yolo',
                             label_fields=('labels',))


# loaded_images = {}

# for idx, row in tqdm.tqdm(selected_images.reset_index().iterrows(),
#                           desc='Loading images into RAM',
#                           total=len(selected_images)):
#     img = cv2.imread(row['filepath'], 0)
#     img = cv2.resize(img, (int(row['width'] * 1024 / row['height']), 1024))

#     loaded_images[row['filepath']] = img


train_images = selected_images.sample(frac=0.85)
train_images


val_images = pd.concat([selected_images, train_images])    .reset_index(
)    .drop_duplicates(keep=False)    .set_index('filepath', drop=True)
val_images


train_images.to_json('yolo/train_images.json', orient='records')
val_images.to_json('yolo/val_images.json', orient='records')


train_annotations = selected_annotations    .reset_index()    .merge(
    train_images.reset_index()['filepath'], on='filepath')    .set_index('bbox_id')
train_annotations


val_annotations = selected_annotations    .reset_index()    .merge(
    val_images.reset_index()['filepath'], on='filepath')    .set_index('bbox_id')
val_annotations


print(len(train_annotations['label'].drop_duplicates()))
print(len(val_annotations['label'].drop_duplicates()))


loaded_images = {}

for img in tqdm.tqdm(selected_images.index,
                     desc='Loading images into RAM',
                     total=len(selected_images)):
    im = cv2.imread(img, 0)
    im = cv2.resize(im, (int(im.shape[1] * 1024 / im.shape[0]), 1024))
    loaded_images[img] = im


weights = 1 / train_annotations[['label', 'filepath']]    .drop_duplicates()    .groupby(
    'label')    .count()    .rename(columns={'filepath': 'weight'})['weight']    .sort_values()

weights


weights = train_annotations[['label', 'filepath']]    .drop_duplicates(
)    .merge(weights, on='label')    .groupby('filepath')    .mean()['weight']
weights


balanced_train_images = weights.sample(
    n=100000, replace=True, weights=weights).index.to_series()
balanced_train_images


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
# path: ../datasets/coco128  # dataset root dir
# train: images/train2017  # train images (relative to 'path') 128 images
# val: images/train2017  # val images (relative to 'path') 128 images
# test:  # test images (optional)

# Classes
# nc: 80  # number of classes
# names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#          'hair drier', 'toothbrush' ]  # class names

Path('yolo2').mkdir(exist_ok=True, parents=True)
Path('yolo2/train/images').mkdir(exist_ok=True, parents=True)
Path('yolo2/train/labels').mkdir(exist_ok=True, parents=True)
Path('yolo2/val/images').mkdir(exist_ok=True, parents=True)
Path('yolo2/val/labels').mkdir(exist_ok=True, parents=True)

Path('yolo2/dataset.yaml').write_text(
    'train: train/images\n'
    'val: val/images\n'
    '\n'
    'nc: 20\n'
    'names: [ ' +
    ', '.join(f"'{lbl}'" for lbl in selected_labels.to_list()) + ' ]\n'

)


aug = alb.Compose(
    transforms=[

        alb.Affine(scale=1,
                   rotate=3,
                   shear=3,
                   mode=cv2.BORDER_REFLECT101,
                   p=0.5),

        alb.HorizontalFlip(p=0.5),

        alb.Equalize(p=0.5),

        alb.RandomGamma(gamma_limit=(85, 115),
                        p=0.5),

    ],
    bbox_params=bbox_params
)


alb_aug = alb.Compose([
    alb.LongestMaxSize(1024),
    alb.PadIfNeeded(1024, 1024, border_mode=cv2.BORDER_CONSTANT, value=255)
], bbox_params=bbox_params)


counter = 0
for idx, img_path in tqdm.tqdm(enumerate(val_images.index),
                               desc='Val images',
                               total=10):
    img = loaded_images[img_path]

    # img = cv2.imread(img_path, 0)
    # img = cv2.resize(img, (int(img.shape[1] * 1024 / img.shape[0]), 1024))

    sel = val_annotations        .loc[val_annotations['filepath'] == img_path].copy(
    )

    bboxes = sel[['x', 'y', 'width', 'height']
                 ].to_records(index=False).tolist()
    labels = sel[['label_id']].to_records(index=False).tolist()

    transformed_sample = alb_aug(image=img, bboxes=bboxes, labels=labels)

    cv2.imwrite(
        f'yolo2/val/images/{counter:05X}.png', transformed_sample['image'])

    ann = ''
    for box, lbl in zip(transformed_sample['bboxes'], transformed_sample['labels']):
        ann += str(lbl[0]) + ' ' + ' '.join([str(b) for b in box])
        ann += '\n'

    Path(f'yolo2/val/labels/{counter:05X}.txt').write_text(ann)

    counter += 1


counter = 0
for idx, img_path in tqdm.tqdm(enumerate(balanced_train_images),
                               desc='Train images',
                               total=len(balanced_train_images)):
    try:
        img = loaded_images[img_path]
        # img = cv2.imread(img_path, 0)
        # img = cv2.resize(img, (int(img.shape[1] * 1024 / img.shape[0]), 1024))

        sel = train_annotations        .loc[train_annotations['filepath'] == img_path].copy(
        )

        bboxes = sel[['x', 'y', 'width', 'height']
                     ].to_records(index=False).tolist()
        labels = sel[['label_id']].to_records(index=False).tolist()

        w = img.shape[1]
        h = img.shape[0]

        cut_lines = np.linspace(0, w, math.ceil(w / h)*2-1)[1:-2]
        cut_lines += np.random.normal(0, math.sqrt(h)/2, len(cut_lines))
        cut_lines = np.array([0, *cut_lines, w-h], dtype=np.int32).clip(0, w-h)

        for cut in cut_lines:
            cropper = alb.Compose([
                alb.Crop(cut, 0, cut+1024, 1024, always_apply=True)],
                bbox_params=bbox_params
            )
            cropped_img = cropper(image=img,
                                  bboxes=bboxes,
                                  labels=labels)
            transformed_sample = aug(**cropped_img)
            # display(transformed_sample)

            cv2.imwrite(
                f'yolo2/train/images/{counter:05X}.png', transformed_sample['image'])

            ann = ''
            for box, lbl in zip(transformed_sample['bboxes'], transformed_sample['labels']):
                ann += str(lbl[0]) + ' ' + ' '.join([str(b) for b in box])
                ann += '\n'

            Path(f'yolo2/train/labels/{counter:05X}.txt').write_text(ann)

            counter += 1

    except Exception as e:
        print(e)
        continue
