from Cargoxray import Cargoxray
import shutil
import os

if __name__ == '__main__':

    dataset = Cargoxray(img_dir='data/v3/images',
                        images_json_path='data/v3/images.json.gz',
                        annotations_json_path='data/v3/annotations.json.gz',
                        categories_json_path='data/v3/categories.json.gz')

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

    selected_labels = [
        'shoes',
        'textile',
        'spare parts',
        'clothes',
        'fabrics',
        'toys',
        'auto parts',
        'tires'
    ]

    dataset.export_data(export_dir='yolo_dataset/',
                        selected_labels=selected_labels,
                        include_empty=True,
                        splits_names=['train', 'val'],
                        splits_frac=[0.8, 0.2],
                        copy_func=os.link)
