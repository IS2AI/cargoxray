from Cargoxray import Cargoxray
import shutil
import os

if __name__ == '__main__':

    dataset = Cargoxray(img_dir='data/v2/images',
                        images_json_path='data/v2/images.json.gz',
                        annotations_json_path='data/v2/annotations.json.gz',
                        categories_json_path='data/v2/categories.json.gz')

    selected_labels = []

    dataset.export_data(export_dir='test_export',
                        selected_labels=selected_labels,
                        include_empty=True,
                        splits_names=['train', 'val'],
                        splits_frac=[0.8, 0.2],
                        copy_func=shutil.copy)
