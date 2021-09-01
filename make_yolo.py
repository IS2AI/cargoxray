import pandas as pd
from pathlib import Path


def run():
    images = pd.read_json('data/images.json.gz',
                          orient='records',
                          compression='gzip')
    annotations = pd.read_json('data/annotations.json.gz',
                               orient='records',
                               compression='gzip')
    
    dataset = annotations.merge(images, 'inner', left_on='image_id', right_on='id')
    print(dataset.iloc[:10].to_string())


if __name__ == '__main__':
    run()
