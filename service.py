import pandas as pd
import json

COLUMNS = ['neighbourhood group', 'instant_bookable', 'cancellation_policy', 'room type', 'Construction year', 'price', 'service fee', 'minimum nights', 'number of reviews', 'review rate number', 'availability 365']

_map_boro_code = {
    'Manhattan': '1',
    'Bronx': '2',
    'Brooklyn': '3',
    'Queens': '4',
    'Staten Island': '5'
}

def convert_price_to_number(price: str):
    price = price.strip()
    if price.startswith('$'):
        price = price[1:]
    
    return int(price.replace(',', ''))

class Service(object):
    data_set = None
    pca = None

    def __init__(self) -> None:
        '''Initializes data'''
        self.init_data()


    def init_data(self) -> None:
        '''Reads data from CSV'''
        data_file = pd.read_csv('nybnb-data.csv')
        self.data_set = data_file[COLUMNS]
        self.__beautify_data()

    def __beautify_data(self) -> None:
        self.data_set['price'] = self.data_set['price'].apply(convert_price_to_number)
        self.data_set['service fee'] = self.data_set['service fee'].apply(convert_price_to_number)
        self.data_set['boro_code'] = self.data_set['neighbourhood group'].apply(lambda n: _map_boro_code.get(n))


if __name__ == '__main__':
    svc = Service()
    svc.init_data()
    f = open('data.json', 'w')
    f.write(svc.data_set.to_json(orient="records"))
    f.close()