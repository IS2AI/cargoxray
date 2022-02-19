python train.py \
--weights yolov5s6.pt \
--data cargoxray/prepared_data/dataset.yaml \
--hyp cargoxray/models/hyp.yaml \
--epochs 100 \
--batch-size 32 \
--imgsz 1024 \
--device 0 \
--project /raid/ruslan_bazhenov/projects/xray/cargoxray/train/new_runs \
--cache

chown 1051:1051 -R /raid/ruslan_bazhenov/projects/xray/cargoxray/train/new_runs