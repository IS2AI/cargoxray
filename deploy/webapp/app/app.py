from re import S
import subprocess
from pathlib import Path
from typing import List, Union

from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont
from shutil import copy, make_archive, rmtree

web_service = Flask(__name__)


@web_service.route('/data', methods=['GET', 'POST'])
def mydata():

    # List of input files urls
    file_urls = request.files.getlist("file[]")

    arc = proc(file_urls)

    # Send the file to user
    resp = send_file(arc)

    # Remove input and output folder, detele zip archive
    Path(arc).unlink()

    # Exit
    return resp


def proc(files: List[Union[str, Path]]) -> Path:

    # Create input and output workind directories
    fld_in = get_next_available_name('images')
    fld_out = Path(f'runs/detect/{fld_in.name}_output')

    fld_in.mkdir()

    # No need to create output folder, YOLO will create it itself
    # Instead, remove it just in case
    rmtree(fld_out)

    # Copy input files into input directory
    for file_url in files:
        copy(file_url, fld_in)

    # Run model with the specified input directory and write
    # results into output directory
    yolo_process = subprocess.Popen(
        f"python detect.py --weights best_yolo_aug.pt "
        f"--source \"{fld_in}\" --img 1024 "
        f"--name \"{fld_out.name}\" --save-txt",
        shell=True)
    yolo_process.wait()

    # Check output images if they have label
    # If not, write "Nothing found" on them

    for output_image in fld_out.iterdir():

        if output_image.is_dir():
            continue

        if not (fld_out / 'labels' / f'{output_image.stem}.txt').exists():
            write_on_image(output_image, "Nothing found")

    # Compress output dir into zip archive
    arc = make_archive(fld_out.name, 'zip', fld_out)

    rmtree(fld_in)
    rmtree(fld_out)

    return Path(arc)


def get_next_available_name(name: Union[str, Path]) -> Path:
    """Find the first available name by appending a number to the initial name"""
    nm = Path(name)

    if not nm.exists():
        return nm

    base_name = nm.name
    i = 0

    while nm.exists():
        nm = nm.parent.joinpath(f'{base_name}{i}')
        i += 1

    return nm


def write_on_image(file_url, message):
    """Writes a message on the image and saves it"""
    im = Image.open(file_url)

    fnt = ImageFont.truetype("Roboto-Regular.ttf", 40)
    drawer = ImageDraw.Draw(im)

    drawer.text((10, 10), message, fill=(255, 0, 0, 255), font=fnt)

    im.save(file_url)


if __name__ == '__main__':
    web_service.run(debug=True, host='0.0.0.0', port=80)
