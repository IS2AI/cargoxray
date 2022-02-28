IMAGES_FRAME_COLUMNS = ['image_id',
                        'file_name',
                        'height',
                        'width',
                        'md5',
                        'size']
IMAGES_FRAME_INDEX = IMAGES_FRAME_COLUMNS[0]

ANNOTATIONS_FRAME_COLUMNS = ['id',
                             'image_id',
                             'category_id',
                             'x',
                             'y',
                             'w',
                             'h']
ANNOTATIONS_FRAME_INDEX = ANNOTATIONS_FRAME_COLUMNS[0]

CATEGORIES_FRAME_COLUMNS = ['category_id',
                            'name']
CATEGORIES_FRAME_INDEX = CATEGORIES_FRAME_COLUMNS[0]

LABEL_REPLACEMENTS_PATH = 'label_mappings_fix.csv'

IMAGE_FORMATS = {'.tif', '.tiff', '.jpg', '.jpeg'}

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10