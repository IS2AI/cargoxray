python src/model/yolov5/val.py --data stages/prepare_cargoxray/dataset.yaml --batch-size 64 --imgsz 1024 --task test --device 0 --project stages --name evaluate --weights stages/train/weights/best.pt