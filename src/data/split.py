from common.Cargoxray import Cargoxray

if __name__ == '__main__':
    dataset = Cargoxray(img_dir='data/images',
                        images_json_path='data/images.json.gz',
                        annotations_json_path='data/annotations.json.gz',
                        categories_json_path='data/categories.json.gz')

    dataset.force_split(train=0.75,
                        val=0.15,
                        test=0.10)

    dataset.apply_changes()
