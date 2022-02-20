from Cargoxray import Cargoxray

if __name__ == '__main__':

    dataset = Cargoxray(img_dir='data/images',
                        images_json_path='data/images.json.gz',
                        annotations_json_path='data/annotations.json.gz',
                        categories_json_path='data/categories.json.gz')

    dataset.import_data(
        import_dir='/raid/ruslan_bazhenov/projects/xray/cargoxray/data/downloads'
    )

    dataset.apply_changes()
