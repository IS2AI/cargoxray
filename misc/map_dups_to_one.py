from pathlib import Path
import json
from hashlib import md5
from typing import Dict, List, Set
import tqdm
import pickle


RAW_DIR = Path('data/raw')


def custom_glob(path, list_of_patterns):
    for pattern in list_of_patterns:
        for filepath in path.glob(pattern):
            yield filepath


def calc_img_hashes(files_dir) -> Dict[Path, bytes]:

    filehashes: Dict[Path, bytes] = {}

    filepath: Path
    for filepath in tqdm.tqdm(list(Path(files_dir).glob('**/*')),
                              'Calculating hashes'):
        if filepath.is_file():
            with filepath.open('rb') as fstream:
                filedata = fstream.read()

            filehashes[filepath] = md5(filedata).digest()
    return filehashes


def get_filehashes(filehashes_path, files_dir, force_rehash) -> Dict[Path, bytes]:

    fhashes_path = Path(filehashes_path)

    if force_rehash or not fhashes_path.exists():
        filehashes = calc_img_hashes(files_dir)
        with fhashes_path.open('wb') as fstream:
            pickle.dump(filehashes, fstream)
    else:
        with fhashes_path.open('rb') as fstream:
            filehashes = pickle.load(fstream)
    return filehashes


def run():
    filehashes = get_filehashes('misc/filehashes.pkl', 'data/raw', False)

    hashtable: Dict[bytes, List[Path]] = {}
    uniques: Dict[str, str] = {}

    filepath: Path
    filehash: bytes
    for filepath, filehash in tqdm.tqdm(filehashes.items(),
                                        'Finding duplicates'):

        with filepath.open('rb') as fstream:
            filedata = fstream.read()

        if filehash in hashtable:
            new_file = True

            for ex_filepath in hashtable[filehash]:
                with ex_filepath.open('rb') as fstream:
                    ex_filedata = fstream.read()
                if ex_filedata == filedata:
                    uniques[filepath.as_posix()] = ex_filepath.as_posix()
                    new_file = False
                    break
            if new_file is True:
                hashtable[filehash].append(filepath)
                uniques[filepath.as_posix()] = filepath.as_posix()
        else:
            hashtable[filehash] = list((filepath,))
            uniques[filepath.as_posix()] = filepath.as_posix()

    with Path('misc/unique_mappings.json').open('w') as fstream:
        json.dump(uniques, fstream, indent=2)
    
    uniques_files = set(uniques.values())
    print(f'There are {len(uniques_files)} unique files')


if __name__ == '__main__':

    run()
