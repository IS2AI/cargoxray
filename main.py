import torch
from cargoxray import CargoXRay
import torchvision


def run():

    ds = CargoXRay('data')

    for img, ann in ds:
        print(img, ann)
        break

if __name__ == '__main__':
    run()
