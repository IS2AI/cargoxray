from pathlib import Path
from typing import List, Union

from PIL import Image, ImageDraw, ImageFont


def proc(output_dir: List[Union[str, Path]]) -> None:

    _output_dir = Path(output_dir)

    for output_image in _output_dir.iterdir():

        if output_image.is_dir():
            continue

        if not (_output_dir / 'labels' / f'{output_image.stem}.txt').exists():
            write_on_image(output_image, "Nothing found")


def write_on_image(file_url, message):
    """Writes a message on the image and saves it"""
    im = Image.open(file_url)

    fnt = ImageFont.truetype("Roboto-Regular.ttf", 40)
    drawer = ImageDraw.Draw(im)

    drawer.text((10, 10), message, fill=(255, 0, 0, 255), font=fnt)

    im.save(file_url)


if __name__ == '__main__':
    proc('runs/detect/exp/')
