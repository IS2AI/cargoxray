from Cargoxray import Cargoxray
import shutil
import os

if __name__ == '__main__':

    dataset = Cargoxray(img_dir='data/v2/images',
                        images_json_path='data/v2/images.json.gz',
                        annotations_json_path='data/v2/annotations.json.gz',
                        categories_json_path='data/v2/categories.json.gz')

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
        # 'tires': 'tires',
        'auto_parts': 'spare_parts'
    }

    # selected_labels = list(selected_labels.keys())

    dataset.export_data(export_dir='data/yolo_slices/merged_notires_auto',
                        selected_labels=selected_labels,
                        include_empty=True,
                        splits_names=['train', 'val'],
                        splits_frac=[0.8, 0.2],
                        copy_func=os.link)
