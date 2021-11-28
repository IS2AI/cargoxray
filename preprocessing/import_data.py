from Cargoxray import Cargoxray

if __name__ == '__main__':

    dataset = Cargoxray(img_dir='data/v2/images',
                        images_json_path='data/v2/images.json.gz',
                        annotations_json_path='data/v2/annotations.json.gz',
                        categories_json_path='data/v2/categories.json.gz')

    dataset.import_data(
        import_dir='/raid/ruslan_bazhenov/projects/xray/cargoxray/data/images',
        empty_images_dir='/raid/ruslan_bazhenov/projects/xray/cargoxray/data/images/empty'
    )

    dataset.apply_changes()
