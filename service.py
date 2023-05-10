import pandas as pd
import json
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances


NUMERICAL_COLUMNS = ['Construction year','price','service fee','minimum nights','number of reviews','review rate number','availability 365']
COLUMNS = ['neighbourhood group', 'instant_bookable', 'cancellation_policy', 'room type', 'Construction year', 'price', 'service fee', 'minimum nights', 'number of reviews', 'review rate number', 'availability 365']

_map_boro_code = {
    'Manhattan': '1',
    'Bronx': '2',
    'Brooklyn': '3',
    'Queens': '4',
    'Staten Island': '5'
}

year_range = ['2003-07', '2008-12', '2013-17', '2018-22']
price_range = ['0-300', '300-600', '600-900', '900-1200']

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
        self._beautify_data()

    def _beautify_data(self) -> None:
        self.data_set['price'] = self.data_set['price'].apply(convert_price_to_number)
        self.data_set['service fee'] = self.data_set['service fee'].apply(convert_price_to_number)
        self.data_set['boro_code'] = self.data_set['neighbourhood group'].apply(lambda n: _map_boro_code.get(n))
        self.data_set['year_range'] = self.data_set['Construction year'].apply(lambda yr: year_range[(yr-2003)//5])
        self.data_set['price_range'] = self.data_set['price'].apply(lambda p: price_range[(p-1)//300])

    def get_mds_data(self):
        df = pd.DataFrame(
            MDS(dissimilarity='precomputed', random_state=49).fit_transform(1-self.data_set[NUMERICAL_COLUMNS].corr()), columns=["x", "y"]
        )
        return self._convert_mds_to_dict(json.loads(df.to_json(orient="records")))
    
    def _convert_mds_to_dict(self, mds_output):
        nodes = [{'id': f'{NUMERICAL_COLUMNS[i]}', 'group': i%2+1} for i in range(len(mds_output))]
        links = []
        for i in range(len(mds_output)):
            for j in range(i+1, len(mds_output)):
                # print(mds_output[i])
                dist = ((mds_output[i]['x'] - mds_output[j]['x'])**2 + (mds_output[i]['y'] - mds_output[j]['y'])**2)**0.5
                links.append({'source': f'{NUMERICAL_COLUMNS[i]}', 'target': f'{NUMERICAL_COLUMNS[j]}', 'value': dist})
        return {'nodes': nodes, 'links': links}


if __name__ == '__main__':
    svc = Service()
    svc.init_data()
    f = open('data.json', 'w')
    print(svc.get_mds_data())
    f.write(svc.data_set.sample(frac=0.01).to_json(orient="records"))
    f.close()