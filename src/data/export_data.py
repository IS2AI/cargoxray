from numpy import require
from common.Cargoxray import Cargoxray
import shutil
import os
import argparse
from pathlib import Path


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    dataset = Cargoxray(img_dir=images_dir,
                        images_json_path=data_dir / 'images.json.gz',
                        annotations_json_path=data_dir / 'annotations.json.gz',
                        categories_json_path=data_dir / 'categories.json.gz')

    # selected_labels = {
    #     'shoes': 'shoes',
    #     'textile': 'textile',
    #     'spare parts': 'auto parts',
    #     'clothes': 'clothes',
    #     'fabrics': 'fabrics',
    #     'toys': 'toys',
    #     'auto parts': 'auto parts',
    #     'tires': 'tires'
    # }

    selected_labels = {
        'shoes': 'shoes',
        'spare_parts': 'spare_parts',
        'clothes': 'clothes',
        'textile': 'textile',
        'toys': 'toys',
        'fabrics': 'fabrics',
        'tires': 'tires',
        'auto_parts': 'spare_parts'
    }

    # selected_labels = list(selected_labels.keys())

    dataset.export_data(export_dir=output_dir,
                        selected_labels=selected_labels,
                        include_empty=True,
                        copy_func=os.link)


if __name__ == '__main__':
    main()
