from pathlib import Path
import json
import shutil
import tqdm
from typing import Dict

DST_DIR = Path('data/images')


def run():

    with Path('misc/unique_mappings.json').open('r') as fs:
        unique_mappings = json.load(fs)

    unique_files = set(unique_mappings.values())

    counter = 0
    copy_mappings: Dict[str, str] = {}

    for filepath in unique_files:
        fp = Path(filepath)

        if fp.suffix == '.tif' \
                or fp.suffix == '.jpg':
            
            new_name = f'{counter:0>5X}{fp.suffix}'
            counter += 1

            copy_mappings[filepath] = DST_DIR.joinpath(new_name).as_posix()


    with Path('misc/copy_mappings.json').open('w') as fs:
        json.dump(copy_mappings, fs)



if __name__ == '__main__':
    run()
