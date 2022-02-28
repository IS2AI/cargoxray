from common.Cargoxray import Cargoxray

if __name__ == '__main__':
    dataset = Cargoxray(img_dir='cargoxray/data/images',
                        images_json_path='cargoxray/data/images.json.gz',
                        annotations_json_path='cargoxray/data/annotations.json.gz',
                        categories_json_path='cargoxray/data/categories.json.gz')

    dataset.force_split(train=0.75,
                        val=0.15,
                        test=0.10)

    dataset.apply_changes()
