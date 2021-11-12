docker build -q -t cargoxray_app .
docker run --rm -it -v "$1:/usr/src/app/input" -v "$2:/usr/src/app/runs/detect/exp/" cargoxray_app