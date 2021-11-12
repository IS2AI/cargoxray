FROM ultralytics/yolov5

WORKDIR /usr/src/app/

ADD app .

RUN pip install -r requirements.txt

EXPOSE 80
# ENTRYPOINT [ "python", "app.py" ]