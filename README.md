# Description
It a dataset of x-ray images of cargo vehicles, such as trucks and railcars.
The project utilizes [YOLOv5](https://github.com/ultralytics/yolov5) model to detect 7 different classes of goods in the x-ray images. YOLOv5 is included in this repository as a git module in `src/model/yolov5`.

# How to run
## Quick start
1. Install dependencies from `requirements.txt`
2. Run `dvc dag` to view the project pipeline
3. Run `dvc repro` to reproduce the experiment.
4. Stages folder will contain the results of executions.

DVC (Data Version Control) manages the project pipeline and automatically rebuilds stages if their dependecies have changed. For example, new images were added, model code has been changed, hyperparameter tuning, etc.

## Details of execution
### Data preparation
1. Import images using `src/data/import_data.py`
- Open the file and set path to dataset destination folder (defaults to `data/cargoxray`)
- Import data by calling `import_data()` method with path to the folder being imported.
2. Run `python src/data/nms.py -i data/cargoxray -o stages/nms_pruning` to filter out duplicate bounding boxes
3. Run `python src/data/export_data.py --images_dir data/cargoxray/images --data_dir stages/nms_pruning --output_dir stages/prepare_cargoxray` to export images and annotations in YOLO format.
### Training
4. Run `python src/model/yolov5/train.py --weights stages/pretrain/weights/best.pt --data stages/prepare_cargoxray/dataset.yaml --hyp params/train_hyp.yaml --epochs 5 --batch-size 64 --imgsz 1024 --device 0,1,2,3 --project stages --name train --cache` to start training
### Inference
6. Run `python src/model/yolov5/detect.py --weights stages/train/weights/best.pt --source path/to/test/images`. Refer to [YOLOv5 detect.py](https://github.com/ultralytics/yolov5/blob/1a2af372d2ae15419a7c2a6ddf4a321d35da38e3/detect.py) usage documentation.

# File descriptions

### `annotations.json.gz`
Stores information on the bounding boxes and their properties 
- `bbox_id`: ID number of a bounding box 
- `image_id`: ID number of an image corresponding to this bounding box
- `x`, `y`, `width`, `height`: Location of the **center** of the bounding box
and its width and height given as fraction of the image size, i.e. from 0 to 1.
- `label`: What is located in this bounding box.

### `images.json.gz`
Stores information on the images and their properties
- `image_id`: ID number of an image
- `size`: Size of the image file in bytes
- `md5`: MD5 checksum of the image file
- `width`, `height`: Dimensions of the image
- `filepath`: Path to the image file

### `json_files.json.gz`
Stores information on the processed JSON files
- `json_id`: ID number of a JSON file
- `filepath`: Path to the JSON file
- `md5`: MD5 checksum of the JSON file
