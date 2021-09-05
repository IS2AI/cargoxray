import json
from pathlib import Path
import shutil
from typing import Dict

RAW_DIR = Path('data/raw')


def run():

    with Path('misc/ann_mappings.json').open('r') as fs:
        ann_mappings = json.load(fs)
    with Path('misc/copy_mappings.json').open('r') as fs:
        copy_mappings = json.load(fs)

    mappings = {}
    for mapp in ann_mappings:
        mappings[mapp] = copy_mappings[ann_mappings[mapp]]

    try:
        with Path('data/annotations.json').open('w') as fs:
            annotations = json.load(fs)
    except FileNotFoundError:
        annotations = []

    for json_path in RAW_DIR.glob('**/*.json'):

        with json_path.open('r') as fstream:
            json_data = json.load(fstream)

        image_info: Dict
        for _, image_info in json_data.items():
            # image_info has {filename, size, regions}
            # regions has {shape_attributes, region_attributes}

            # list of images
            #     id, filepath, size, etc.
            # list of annotations
            #     id, image_id, shape_attributes, label

            # annotations
            #  [
            # ]
            try:
                new_fp = mappings[image_info['filename']+str(image_info['size'])]
            except KeyError:
                new_fp = image_info['filename']+str(image_info['size'])

            annotations.append({
                'image': {
                    'filepath': new_fp,
                    'size': image_info['size'],
                    'height': 0,
                    'width': 0,
                    'md5': '',
                },
                'regions': image_info['regions'] if isinstance(image_info['regions'], list) else list(image_info['regions'].values())
            })
        json_path.unlink()

    print(len(annotations))

    with Path('data/annotations.json').open('w') as fs:
        json.dump(annotations, fs, indent=2)


if __name__ == '__main__':
    run()
