# Description
It a dataset of x-ray images of cargo vehicles, such as trucks and railcars. Annotations are in YOLO format stored in JSON files with Pandas.

# File description
annotations.json.gz
- bbox_id
- image_id
- x
- y
- width
- height
- label

`images.json.gz`
- `image_id`
- `size`
- `md5`
- `width`
- `height`
- `filepath`

- [ ] z
- [x] he

`json_files.json.gz`
- `json_id`
- `filepath`
- `md5`


## Data directory
D

## Utils
Utils contain scripts for importing new images and generating YOLO dataset