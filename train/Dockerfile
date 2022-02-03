FROM ultralytics/yolov5

ADD Arial.ttf /usr/src/app/.config/Ultralytics/
ADD .netrc /usr/src/app/

ADD hyp.yaml /usr/src/app/
ADD train.sh /usr/src/app/

ENTRYPOINT [ "sh", "train.sh"]