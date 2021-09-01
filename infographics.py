from numpy import nan
import pandas as pd


def run():
    images = pd.read_json('data/images.json.gz',
                          orient='records',
                          compression='gzip')
    annotations = pd.read_json('data/annotations.json.gz',
                               orient='records',
                               compression='gzip')

    working_anns = annotations.loc[
        annotations['image_id'].notna()
        & annotations['label'].notna()
    ].sort_values('image_id')

    print(working_anns)
    print(len(working_anns['image_id'].unique()))

    working_anns = working_anns.replace((
        'brokkoli',
        'clouthes',
        'equpment',
        'grapes',
        'motobike',
        'motorcycle',
        'nectarine',
        'pears',
        'scooters',
        'textiles',
        'tomates'
    ), (
        'broccoli',
        'clothes',
        'equipment',
        'grape',
        'bike',
        'bike',
        'nectarin',
        'pear',
        'scooter',
        'textile',
        'tomato'
    ))
    print(working_anns[['id', 'label']]
          .groupby('label')
          .count()
          .sort_values('id', ascending=False)
          .to_string()
          )


if __name__ == '__main__':
    run()
