FROM ultralytics/yolov5:v5.0

ADD hyp.yaml /usr/src/app/
ADD dataset.yaml /usr/src/app/
ADD Arial.ttf /usr/src/app/.config/Ultralytics/
ADD .netrc /usr/src/app/

# ENTRYPOINT [ "nvidia-smi" ]

ENTRYPOINT [ \
    "taskset", "--cpu-list", "21,22,23,24,25,26,27,28,29,30,31,32,33,34,35", \
    "python", "train.py", \
    "--cfg", "yolov5s6.yaml", \
    "--img", "1024", \
    "--data", "dataset.yaml", \
    "--hyp", "hyp.yaml", \
    "--epochs", "80", \
    "--batch-size", "32", \
    "--cache", \
    "--workers", "8", \
    # "--patience", "5", \
    "--device=0", \
    "--project", "/raid/ruslan_bazhenov/projects/xray/cargoxray/train/new_runs" \
]