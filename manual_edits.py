from numpy import nan
import pandas as pd


def run():
    annotations = pd.read_json('data/annotations.json',
                               orient='records')

    annotations = annotations.replace((
        'brokkoli',
        'clouthes',
        'equpment',
        'Equipment',
        'grapes',
        'motobike',
        'motorcycle',
        'nectarine',
        'pears',
        'scooters',
        'textiles',
        'tomates',
        'Household goods',
        'Lamps'
    ), (
        'broccoli',
        'clothes',
        'equipment',
        'equipment',
        'grape',
        'bike',
        'bike',
        'nectarin',
        'pear',
        'scooter',
        'textile',
        'tomato',
        'household goods',
        'lamps'
    ))

    annotations.to_json('data/annotations.json',
                        orient='records',
                        indent=2)

if __name__ == '__main__':
    run()