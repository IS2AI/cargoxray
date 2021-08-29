from pathlib import Path
import json
import shutil
import tqdm
from typing import Dict

DST_DIR = Path('data/images')


def run():

    
    with Path('misc/copy_mappings.json').open('r') as fs:
        copy_mappings = json.load(fs)

    DST_DIR.mkdir(exist_ok=True)
    for src, dst in tqdm.tqdm(copy_mappings.items(),
                              total=len(copy_mappings),
                              desc='Copying images'):
        shutil.copy(src, dst)
        Path(src).unlink()


if __name__ == '__main__':
    run()
