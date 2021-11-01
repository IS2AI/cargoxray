import argparse

PROGRAM_DESCRIPTION = 'Parses the dataset to extract images and annotations in YOLO format'


def run():
    pass


def parse_args():

    argparser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION)

    path_args = argparser.add_argument_group('Paths')
    classes_args = argparser.add_argument_group('Selected classes')
    cfg_args = argparser.add_argument_group('Config file')

    path_args.add_argument('--root',
                           action='store',
                           default='data/',
                           help='Other paths, including those are in JSON files, are relative to this.',
                           required=True)
    path_args.add_argument('--annotations',
                           action='store',
                           default='annotations.json.gz',
                           help='Path to annotations JSON file',
                           required=True)
    path_args.add_argument('--images',
                           action='store',
                           default='images.json.gz',
                           help='Path to images JSON file',
                           required=True)
    path_args.add_argument('--output',
                           action='store',
                           default='yolo_dataset',
                           help='Path to the output directory',
                           required=True)
    classes_args.add_argument('-l',
                              '--selected_classes',
                              action='store',
                              nargs='+',
                              default=[],
                              metavar=('CLASS 1', 'CLASS 2'),
                              help='Selected list of classes to extract')
    classes_args.add_argument('-u',
                              '--unlabeled',
                              action='store_true',
                              help='Include unlabeled bounding boxes as "other" superclass')
    classes_args.add_argument('-o',
                              '--other',
                              action='store_true',
                              help='Include ignored classes as "other" superclass')
    classes_args.add_argument('-e',
                              '--empty',
                              action='store_true',
                              help='Include empty images')
    cfg_args.add_argument('-c',
                          '--cfg',
                          action='store',
                          metavar='CONFIG_FILE')

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(args.selected_classes)
