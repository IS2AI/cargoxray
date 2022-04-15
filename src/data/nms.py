# import the necessary packages
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    return args


def non_max_suppression_fast(boxes, overlapThresh):

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    # return boxes[pick].astype("int")
    return boxes[pick, 4].astype("int")


def proc(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reset_index().copy()
    frame['xx'] = frame['x'] + frame['w']
    frame['yy'] = frame['y'] + frame['h']
    # print(frame)
    # print(frame[['x', 'y', 'xx', 'yy', 'id']].to_numpy())
    pick = non_max_suppression_fast(
        frame[['x', 'y', 'xx', 'yy', 'id']].to_numpy(), 0.90)

    frame = frame.drop(columns=['xx', 'yy']).set_index('id')
    pick = frame.loc[pick]

    return pick


def main():

    args = parse_args()

    data_dir = Path(args.input)
    output_dir = Path(args.output)

    # data_dir = Path(
    #     '/raid/ruslan_bazhenov/projects/xray/cargoxray/data/cargoxray')
    # output_dir = Path('/raid/ruslan_bazhenov/projects/xray/cargoxray/stages/nms_pruning')

    output_dir.mkdir(parents=True)

    annotations: pd.DataFrame = pd.read_json(data_dir / 'annotations.json.gz',
                                             orient='records',
                                             compression='gzip')\
        .set_index(['image_id', 'id'])

    res = pd.DataFrame()

    for k in tqdm.tqdm(annotations.index.get_level_values(0).drop_duplicates()):
        r = proc(annotations.loc[k])
        r['image_id'] = k

        res = pd.concat([res, r])

    res = res[['image_id', 'category_id', 'x', 'y', 'w', 'h']]

    res.reset_index().to_json(output_dir / 'annotations.json.gz',
                orient='records',
                compression='gzip')

    shutil.copy2(data_dir / 'images.json.gz', output_dir)
    shutil.copy2(data_dir / 'categories.json.gz', output_dir)


if __name__ == '__main__':
    main()
