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
        'Lamps',
        'Car wheels',
        'Clothes',
        'Shoes',
        'Spare parts',
        'appliances',
        'car wheels ',
        'carweels',
        'carwheels',
        'cars',
        'equipment '
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
        'lamps',
        'car wheels',
        'clothes',
        'shoes',
        'spare parts',
        'appliance',
        'car wheels',
        'car wheels',
        'car wheels',
        'car',
        'equipment'
    ))

    for exx in annotations['label'].drop_duplicates().sort_values().to_list():
        print("\"" + exx + "\"")

    annotations.to_json('data/annotations.json',
                        orient='records',
                        indent=2)

if __name__ == '__main__':
    run()