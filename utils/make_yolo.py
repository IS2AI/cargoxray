# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


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
import math


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
annotations = annotations    .reset_index()    .merge(images[['old_location', 'width', 'height']],
           on='image_id',
           suffixes=('_bbox', '_image'))\
    .drop_duplicates(['bbox_id'])\
    .set_index('bbox_id')\
    .rename(columns={'old_location': 'filepath'})
annotations


# %%

annotations = annotations.assign(width=annotations['width_bbox'] / annotations['width_image'],
                   height=annotations['height_bbox'] / annotations['height_image'],
                   x=(annotations['x_min']+annotations['width_bbox'] / 2) / annotations['width_image'],
                   y=(annotations['y_min']+annotations['height_bbox'] / 2) / annotations['height_image'],)\
                       [['label','x', 'y', 'width', 'height', 'filepath']]


# %%
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
selected_labels = labels.iloc[:20]
selected_labels = selected_labels.reset_index()['label']
selected_labels.index.name = 'label_id'
selected_labels


# %%

', '.join(f"'{lbl}'" for lbl in selected_labels.to_list())


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
selected_labels.to_frame().reset_index()


# %%
selected_annotations = pd.concat([annotations, ignored_annotations])    .drop_duplicates(keep=False)

selected_annotations = selected_annotations.reset_index().merge(selected_labels.to_frame(
).reset_index()).set_index('bbox_id')[['label', 'label_id', 'x', 'y', 'width', 'height', 'filepath']]

selected_annotations


# %%
selected_images = selected_annotations['filepath']    .drop_duplicates()    .reset_index(drop=True)
selected_images


# %%
selected_images = images[['old_location', 'width', 'height']]    .rename(columns={'old_location': 'filepath'})    .drop_duplicates('filepath')    .merge(selected_images, on='filepath')    .set_index('filepath', drop=True)
selected_images


# %%
bbox_params = alb.BboxParams(format='yolo',
                             label_fields=('labels',))


# %%
# loaded_images = {}

# for idx, row in tqdm.tqdm(selected_images.reset_index().iterrows(),
#                           desc='Loading images into RAM',
#                           total=len(selected_images)):
#     img = cv2.imread(row['filepath'], 0)
#     img = cv2.resize(img, (int(row['width'] * 1024 / row['height']), 1024))

#     loaded_images[row['filepath']] = img


# %%
train_images = selected_images.sample(frac=0.85)
train_images


# %%
val_images = pd.concat([selected_images, train_images])    .reset_index()    .drop_duplicates(keep=False)    .set_index('filepath', drop=True)
val_images


# %%
train_images.to_json('yolo/train_images.json', orient='records')
val_images.to_json('yolo/val_images.json', orient='records')


# %%
train_annotations = selected_annotations    .reset_index()    .merge(train_images.reset_index()['filepath'], on='filepath')    .set_index('bbox_id')
train_annotations


# %%
val_annotations = selected_annotations    .reset_index()    .merge(val_images.reset_index()['filepath'], on='filepath')    .set_index('bbox_id')
val_annotations


# %%
print(len(train_annotations['label'].drop_duplicates()))
print(len(val_annotations['label'].drop_duplicates()))


# %%
loaded_images = {}

for img in tqdm.tqdm(selected_images.index,
                     desc='Loading images into RAM',
                     total=len(selected_images)):
    im = cv2.imread(img, 0)
    im = cv2.resize(im, (int(im.shape[1] * 1024 / im.shape[0]), 1024))
    loaded_images[img] = im


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
    'names: [ ' + ', '.join(f"'{lbl}'" for lbl in selected_labels.to_list()) + ' ]\n'
    
)


# %%
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


# %%
alb_aug = alb.Compose([
    alb.LongestMaxSize(1024),
    alb.PadIfNeeded(1024, 1024, border_mode=cv2.BORDER_CONSTANT,value=255)
], bbox_params=bbox_params)


# %%
counter = 0
for idx, img_path in tqdm.tqdm(enumerate(val_images.index),
                               desc='Val images',
                               total=10):
    img = loaded_images[img_path]

    # img = cv2.imread(img_path, 0)
    # img = cv2.resize(img, (int(img.shape[1] * 1024 / img.shape[0]), 1024))

    sel = val_annotations        .loc[val_annotations['filepath'] == img_path].copy()

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


# %%
counter = 0
for idx, img_path in tqdm.tqdm(enumerate(balanced_train_images),
                               desc='Train images',
                               total=len(balanced_train_images)):
    try:
        img = loaded_images[img_path]
        # img = cv2.imread(img_path, 0)
        # img = cv2.resize(img, (int(img.shape[1] * 1024 / img.shape[0]), 1024))

        sel = train_annotations        .loc[train_annotations['filepath'] == img_path].copy()

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


# %%
# for idx, img_loc in tqdm.tqdm(enumerate(balanced_train_images),
#                               total=len(balanced_train_images)):

#     image = Image.open(img_loc)
#     if image.mode == 'I;16':
#         image = image.convert('I').point(
#             [i/256 for i in range(2**16)], 'L')
#     else:
#         image = image.convert('L')

#     sel = train_annotations.loc[train_annotations['filepath'] == img_loc].copy(
#     )

#     sel['x'] = sel['x_min'] + sel['width'] // 2
#     sel['y'] = sel['y_min'] + sel['height'] // 2

#     sel['x'] /= image.width
#     sel['y'] /= image.height
#     sel['width'] /= image.width
#     sel['height'] /= image.height

#     bboxes = sel[['x', 'y', 'width', 'height']
#                  ].to_records(index=False).tolist()
#     labels = [selected_labels.index[selected_labels == lbl][0]
#               for lbl in sel['label'].tolist()]

#     sample = {
#         'image': np.asarray(image),
#         'bboxes': bboxes,
#         'labels': labels,
#     }

#     image.close()
#     try:

#         transformed_sample = aug(**sample)
#     except:
#         print(img_loc)
#         continue

#     image = Image.fromarray(transformed_sample['image'])
#     image.save(f'yolo/train/images/{idx:05X}.png')
#     image.close()

#     bboxes = transformed_sample['bboxes']
#     labels = transformed_sample['labels']

#     bboxes = '\n'.join([
#         f'{lbl} {box[0]} {box[1]} {box[2]} {box[3]}' for box, lbl in zip(bboxes, labels)]
#     )

#     Path(f'yolo/train/labels/{idx:05X}.txt').write_text(bboxes)


#     # bboxes = [(int((x-w/2)*500),
#     #            int((y-h/2)*500),
#     #            int((x+w/2)*500),
#     #            int((y+h/2)*500),) for x, y, w, h, in transformed_sample['bboxes']]

#     # wbboxes = tf.ToPILImage()(torchvision.utils.draw_bounding_boxes(
#     #     torch.from_numpy(transformed_sample['image']).expand(3, -1, -1),
#     #     boxes=torch.IntTensor(bboxes),
#     #     labels=[str(l) for l in transformed_sample['labels']],
#     #     fill=True,
#     #     colors=['red'] * len(bboxes),
#     # ))
    
#     # display(wbboxes)


# %%



