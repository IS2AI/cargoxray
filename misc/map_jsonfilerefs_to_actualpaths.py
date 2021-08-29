from pathlib import Path
import json
from hashlib import md5
from typing import Dict, List, Set
import tqdm
import pickle

def run():

    with Path('misc/unique_mappings.json').open('r') as fs:
        unique_mappings = json.load(fs)
    
    ann_mappings: Dict[str, str] = {}

    for filepath, alias in unique_mappings.items():
        
        fp = Path(filepath)

        if fp.suffix == '.tif' \
            or fp.suffix == '.jpg':
        
            ref = f'{fp.name}{fp.stat().st_size}'
            if ref in ann_mappings:
                if ann_mappings[ref] != alias:
                    raise KeyError(alias, ann_mappings[ref])
            else:
                ann_mappings[ref] = alias

    with Path('misc/ann_mappings.json').open('w') as fs:
        json.dump(ann_mappings, fs, indent=2)


if __name__ == '__main__':
    run()