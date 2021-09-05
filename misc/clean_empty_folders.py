from pathlib import Path
import shutil


def cleanup(folder: Path):
    if not folder.is_dir():
        return

    print(folder.as_posix())

    for dir in folder.iterdir():
        cleanup(dir)

        if dir.is_file():
            if dir.name == '.DS_Store' or dir.suffix == '.tif' or dir.suffix == '.jpg' or dir.name == 'Thumbs.db':
                dir.unlink()
    
    try:
        folder.rmdir()
    except OSError:
        pass

def run():
    
    root = Path('data/raw')

    cleanup(root)


if __name__ == '__main__':
    run()