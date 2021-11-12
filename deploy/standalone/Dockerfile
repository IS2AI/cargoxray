FROM ultralytics/yolov5

WORKDIR /usr/src/app/

ADD app .
ADD app/Arial.ttf /usr/src/app/.config/Ultralytics/Arial.ttf

ENTRYPOINT [ "sh", "run.sh" ]