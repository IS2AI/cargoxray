# %%
import shutil
from pathlib import Path

import albumentations as alb
import cv2
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torchvision import transforms as tf
import torch
import tqdm


# %%
images = pd.read_json('data_new/images.json.gz',
                        orient='records',
                        typ='frame',
                        compression='gzip')
images = images.set_index('image_id')
images


# %%
annotations = pd.read_json('data_new/annotations.json.gz',
                            orient='records',
                            typ='frame',
                            compression='gzip')
annotations = annotations.set_index('bbox_id')
annotations


# %%
annotations = annotations    .reset_index()    .merge(images['old_location'],
           on='image_id')\
    .drop_duplicates(['bbox_id',
                      'image_id'])\
    .set_index('bbox_id')\
    .rename(columns={'old_location': 'filepath'})
annotations


# %%
annotations = annotations.replace((
        'brokkoli',
        'clouthes',
        'equpment',
        'Equipment',
        'grapes',
        'motobike',
        'motorcycle',
        'nectarine',
        'pears',
        'scooters',
        'textiles',
        'tomates',
        'Household goods',
        'Lamps',
        'Car wheels',
        'Clothes',
        'Shoes',
        'Spare parts',
        'appliances',
        'car wheels ',
        'carweels',
        'carwheels',
        'cars',
        'equipment '
    ), (
        'broccoli',
        'clothes',
        'equipment',
        'equipment',
        'grape',
        'bike',
        'bike',
        'nectarin',
        'pear',
        'scooter',
        'textile',
        'tomato',
        'household goods',
        'lamps',
        'car wheels',
        'clothes',
        'shoes',
        'spare parts',
        'appliance',
        'car wheels',
        'car wheels',
        'car wheels',
        'car',
        'equipment'
    ))


# %%
labels = annotations[['label', 'filepath']]    .drop_duplicates()    .groupby('label')    .count()['filepath']    .sort_values(ascending=False)


labels.name = 'labels'
ignored_labels = labels.iloc[20:]
ignored_labels


# %%
ignored_annotations = annotations    .reset_index()    .merge(ignored_labels,
           on='label')\
    .drop(columns=['labels'])\
    .set_index('bbox_id')

ignored_annotations


# %%
ignored_images = ignored_annotations['filepath'].drop_duplicates()
ignored_images


# %%
ignored_annotations = annotations    .reset_index()    .merge(ignored_images,
           on='filepath')\
    .set_index('bbox_id')
ignored_annotations


# %%
selected_annotations = pd.concat([annotations, ignored_annotations])    .drop_duplicates(keep=False)

selected_annotations


# %%
selected_images = selected_annotations['filepath']    .drop_duplicates()    .reset_index(drop=True)
selected_images


# %%
selected_labels = selected_annotations['label']    .drop_duplicates()    .reset_index(drop=True)
    
selected_labels.index.name = 'label_id'

', '.join(f"'{lbl}'" for lbl in selected_labels.to_list())


# %%
train_images = selected_images.sample(frac=0.85)
train_images


# %%
val_images = pd.concat([selected_images, train_images])    .drop_duplicates(keep=False)
val_images


# %%

train_images = pd.read_json('yolo/train_images.json', typ='series',orient='records')
val_images = pd.read_json('yolo/val_images.json', typ='series', orient='records')

train_images.name = 'filepath'
val_images.name = 'filepath'


# %%
train_annotations = selected_annotations    .reset_index()    .merge(train_images, on='filepath')    .set_index('bbox_id')
train_annotations


# %%
val_annotations = selected_annotations    .reset_index()    .merge(val_images, on='filepath')    .set_index('bbox_id')
val_annotations


# %%
print(len(train_annotations['label'].drop_duplicates()))
print(len(val_annotations['label'].drop_duplicates()))


# %%
weights = 1 / train_annotations[['label', 'filepath']]    .drop_duplicates()    .groupby('label')    .count()    .rename(columns={'filepath': 'weight'})['weight']    .sort_values()

weights


# %%
weights = train_annotations[['label', 'filepath']]    .drop_duplicates()    .merge(weights,on='label')    .groupby('filepath')    .mean()['weight']
weights


# %%

balanced_train_images = weights.sample(n=100000, replace=True, weights=weights).index.to_series()
balanced_train_images


# %%
bbox_params = alb.BboxParams(format='yolo',
                             label_fields=('labels',))

aug = alb.Compose(
    transforms=[

        alb.SmallestMaxSize(max_size=600),

        alb.Affine(scale=1,
                   rotate=3,
                   shear=3,
                   mode=cv2.BORDER_REFLECT101,
                   p=0.25),

        alb.SmallestMaxSize(max_size=500),

        alb.RandomCrop(500, 500),

        alb.HorizontalFlip(p=0.25),

        alb.Equalize(p=0.25),

        alb.RandomGamma(gamma_limit=(85, 115),
                        p=0.25),

    ],
    bbox_params=bbox_params
)


# %%
# # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
# path: ../datasets/coco128  # dataset root dir
# train: images/train2017  # train images (relative to 'path') 128 images
# val: images/train2017  # val images (relative to 'path') 128 images
# test:  # test images (optional)

# # Classes
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


Path('yolo/dataset.yaml').write_text(
    'train: train/images\n'
    'val: val/images\n'
    '\n'
    'nc: 20\n'
    'names: [ ' + ', '.join(f"'{lbl}'" for lbl in selected_labels.to_list()) + ' ]\n'
    
)


# %%
aug_val = alb.Compose(
    transforms=[
        alb.SmallestMaxSize(max_size=500),
    ],
    bbox_params=bbox_params
)

# %%


for idx, img_loc in tqdm.tqdm(enumerate(val_images),
                              total=len(val_images)):

    image = Image.open(img_loc)
    if image.mode == 'I;16':
        image = image.convert('I').point(
            [i/256 for i in range(2**16)], 'L')
    else:
        image = image.convert('L')

    sel = val_annotations.loc[val_annotations['filepath'] == img_loc].copy()

    sel['x'] = sel['x_min'] + sel['width'] // 2
    sel['y'] = sel['y_min'] + sel['height'] // 2

    sel['x'] /= image.width
    sel['y'] /= image.height
    sel['width'] /= image.width
    sel['height'] /= image.height

    bboxes = sel[['x', 'y', 'width', 'height']
                 ].to_records(index=False).tolist()
    labels = [selected_labels.index[selected_labels == lbl][0]
              for lbl in sel['label'].tolist()]

    sample = {
        'image': np.asarray(image),
        'bboxes': bboxes,
        'labels': labels,
    }

    image.close()
    try:

        transformed_sample = aug_val(**sample)
    except:
        print(img_loc)
        continue

    image = Image.fromarray(transformed_sample['image'])
    image.save(f'yolo/val/images/{idx:05X}.png')
    image.close()

    bboxes = transformed_sample['bboxes']
    labels = transformed_sample['labels']

    bboxes = '\n'.join([
        f'{lbl} {box[0]} {box[1]} {box[2]} {box[3]}' for box, lbl in zip(bboxes, labels)]
    )

    Path(f'yolo/val/labels/{idx:05X}.txt').write_text(bboxes)


# %%
for idx, img_loc in tqdm.tqdm(enumerate(balanced_train_images),
                              total=len(balanced_train_images)):

    image = Image.open(img_loc)
    if image.mode == 'I;16':
        image = image.convert('I').point(
            [i/256 for i in range(2**16)], 'L')
    else:
        image = image.convert('L')

    sel = train_annotations.loc[train_annotations['filepath'] == img_loc].copy(
    )

    sel['x'] = sel['x_min'] + sel['width'] // 2
    sel['y'] = sel['y_min'] + sel['height'] // 2

    sel['x'] /= image.width
    sel['y'] /= image.height
    sel['width'] /= image.width
    sel['height'] /= image.height

    bboxes = sel[['x', 'y', 'width', 'height']
                 ].to_records(index=False).tolist()
    labels = [selected_labels.index[selected_labels == lbl][0]
              for lbl in sel['label'].tolist()]

    sample = {
        'image': np.asarray(image),
        'bboxes': bboxes,
        'labels': labels,
    }

    image.close()
    try:

        transformed_sample = aug(**sample)
    except:
        print(img_loc)
        continue

    image = Image.fromarray(transformed_sample['image'])
    image.save(f'yolo/train/images/{idx:05X}.png')
    image.close()

    bboxes = transformed_sample['bboxes']
    labels = transformed_sample['labels']

    bboxes = '\n'.join([
        f'{lbl} {box[0]} {box[1]} {box[2]} {box[3]}' for box, lbl in zip(bboxes, labels)]
    )

    Path(f'yolo/train/labels/{idx:05X}.txt').write_text(bboxes)


    # bboxes = [(int((x-w/2)*500),
    #            int((y-h/2)*500),
    #            int((x+w/2)*500),
    #            int((y+h/2)*500),) for x, y, w, h, in transformed_sample['bboxes']]

    # wbboxes = tf.ToPILImage()(torchvision.utils.draw_bounding_boxes(
    #     torch.from_numpy(transformed_sample['image']).expand(3, -1, -1),
    #     boxes=torch.IntTensor(bboxes),
    #     labels=[str(l) for l in transformed_sample['labels']],
    #     fill=True,
    #     colors=['red'] * len(bboxes),
    # ))
    
    # display(wbboxes)