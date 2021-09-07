import pandas as pd
from pathlib import Path
import shutil
import random


def run():
    # images = pd.read_json('data/images.json',
    #                       orient='records')
    # annotations = pd.read_json('data/annotations.json',
    #                            orient='records')
    # Path('data/labels').mkdir(parents=True, exist_ok=True)

    # dataset: pd.DataFrame
    # dataset = annotations.merge(
    #     images, 'inner', left_on='image_id', right_on='id')
    # dataset = dataset.drop('id_y', axis='columns').rename(
    #     {'id_x': 'id'}, axis='columns')
    # dataset = dataset[['id', 'filepath', 'x_points', 'y_points', 'label']]
    # dataset = dataset.sort_values('id')

    # # Select interesting only

    # sel = dataset \
    #     .drop_duplicates(['label', 'filepath']) \
    #     .groupby('label') \
    #     .count()['id']
    # sel = sel.sort_values(ascending=False)
    # sel = sel.iloc[:10]

    # labels = sel.index.values

    # labels = {val: idx for idx, val in enumerate(labels)}

    # with Path('data/yolo/yolo.yaml').open('w') as fs:
    #     fs.write('train: data/yolo/images\n')
    #     fs.write('val: data/yolo/images\n')
    #     fs.write('nc: {}\n'.format(len(labels)))
    #     fs.write(
    #         f'names: [{", ".join([str(l) for l in labels.keys()])}]\n')

    # for img in dataset.drop_duplicates('filepath')['filepath']:
    #     print(img)

    #     yolo_ann_path = Path(f"data/yolo/labels/{Path(img).stem}.txt")

    #     yolo_ann_path.unlink(missing_ok=True)

    #     img_w = None
    #     img_h = None

    #     for idx, ann in dataset.loc[dataset['filepath'] == img].iterrows():

    #         if ann['label'] in labels:

    #             if(img_w is None or img_h is None):

    #                 with PIL.Image.open(img) as im:
    #                     img_w, img_h = im.size

    #                 dst = Path('data/yolo/images', Path(img).name)

    #                 shutil.copy(img, dst)

    #             x = pd.Series(ann['x_points'])
    #             y = pd.Series(ann['y_points'])
    #             w = x.max() - x.min()
    #             h = y.max() - y.min()
    #             x = (x.max() + x.min()) // 2
    #             y = (y.max() + y.min()) // 2

    #             x /= img_w
    #             y /= img_h
    #             w /= img_w
    #             h /= img_h

    #             with yolo_ann_path.open('a', encoding='utf-8') as fs:
    #                 print(labels[ann['label']], x, y, w, h, file=fs)

    th = 0.8
    for img in Path('data/yolo/images').glob('*.jpg'):
        if random.random() > th:
            shutil.move(img,
                        'data/yolo/images/val/')
            shutil.move(f'data/yolo/labels/{img.stem}.txt',
                        'data/yolo/labels/val/')
        else:
            shutil.move(img,
                        'data/yolo/images/train/')
            shutil.move(f'data/yolo/labels/{img.stem}.txt',
                        'data/yolo/labels/train/')


if __name__ == '__main__':
    run()
