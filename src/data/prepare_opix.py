from pathlib import Path
import shutil

from PIL import Image

OPIX_PATH = 'opix_dataset'
OPIX_PREPARED = 'prepared_opix'

TO_GRAYSCALE = True


def prepare(dir, dst):
    dir = Path(dir)
    dst = Path(dst)

    for p in dir.glob('**/*'):

        if p.is_dir():
            continue
        
        d = dst / p.relative_to(dir)
        d.parent.mkdir(parents=True, exist_ok=True)

        if p.suffix == '.jpg' and TO_GRAYSCALE:
            im = Image.open(p)
            im = im.convert('L')
            im.save(d)
        else:
            shutil.copy(p, d)

        print(d)


if __name__ == '__main__':
    prepare(OPIX_PATH, OPIX_PREPARED)
