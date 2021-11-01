# Description
It a dataset of x-ray images of cargo vehicles, such as trucks and railcars.
Annotations are in YOLO format stored in JSON files with Pandas.

# File descriptions

### annotations.json.gz
Stores information on the bounding boxes and their properties 
- `bbox_id`: ID number of a bounding box 
- `image_id`: ID number of an image corresponding to this bounding box
- `x`, `y`, `width`, `height`: Location of the **center** of the bounding box
and its width and height given as fraction of the image size, i.e. from 0 to 1.
- `label`: What is located in this bounding box.

### images.json.gz
Stores information on the images and their properties
- `image_id`: ID number of an image
- `size`: Size of the image file in bytes
- `md5`: MD5 checksum of the image file
- `width`, `height`: Dimensions of the image
- `filepath`: Path to the image file

### json_files.json.gz
Stores information on the processed JSON files
- `json_id`: ID number of a JSON file
- `filepath`: Path to the JSON file
- `md5`: MD5 checksum of the JSON file


## Utils
Utils contain scripts for importing new images and generating YOLO datasets.

# TODO
- [ ] Detect duplicate annotations
- [ ] Make a report after images import/export
- [x] Run import/export script with command line arguments