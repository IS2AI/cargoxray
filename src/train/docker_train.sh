docker build . -t cargoxray_train
docker run --rm -it --name cargoxray_train --runtime=nvidia -v /raid/ruslan_bazhenov:/raid/ruslan_bazhenov --shm-size=10g --gpus device=11 cargoxray_train