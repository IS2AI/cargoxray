docker run --rm --name cargoxray_train --runtime=nvidia -v /raid/ruslan_bazhenov:/raid/ruslan_bazhenov --shm-size=10g --gpus device=0 loyaltytwo/cargoxray_train

docker-compose up