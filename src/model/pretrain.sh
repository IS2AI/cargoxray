python src/model/yolov5/train.py --weights /usr/src/basemodels/yolov5s6.pt
      --data prepared_opix/dataset.yaml --hyp models/params/pretrain_hyp.yaml --epochs
      1 --batch-size 64 --imgsz 1024 --device 0 --project models --name pretrained
      --cache