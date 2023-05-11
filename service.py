import pandas as pd
import json
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances


NUMERICAL_COLUMNS = ['review rate number','Construction year','price','service fee','minimum nights','number of reviews','availability 365']
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
    map_room_type = {
        'Entire_home_apt': "Entire home/apt",
        'Private_room': "Private room",
        'Shared_room': "Shared room",
        'Hotel_room': "Hotel room"
    }

    def __init__(self) -> None:
        '''Initializes data'''
        self.init_data()


    def init_data(self) -> None:
        '''Reads data from CSV'''
        data_file = pd.read_csv('nybnb-data.csv')
        self.data_set = data_file[COLUMNS].sample(frac=0.01)
        self._beautify_data()

    def _beautify_data(self) -> None:
        self.data_set['price'] = self.data_set['price'].apply(convert_price_to_number)
        self.data_set['service fee'] = self.data_set['service fee'].apply(convert_price_to_number)
        self.data_set['boro_code'] = self.data_set['neighbourhood group'].apply(lambda n: _map_boro_code.get(n))
        self.data_set['year_range'] = self.data_set['Construction year'].apply(lambda yr: year_range[(yr-2003)//5])
        self.data_set['price_range'] = self.data_set['price'].apply(lambda p: price_range[(p-1)//300])

    
    # {'MAP': ['4', '6'], 'SCATTER_PLOT': [4], 'BAR_GRAPH': [{'key': '2018-22', 'Entire_home_apt': 98, 'Hotel_room': 1, 'Private_room': 75, 'Shared_room': 1, 'roomType': 'Private_room'}], 'SUNBURST': [{'0': '900-1200', '1': False, '2': 'strict'}]}
    def get_filtered_data(self, filters):
        all_data = json.loads(self.data_set.to_json(orient="records"))
        data = list(filter(lambda d: self.apply_filter(d, filters), all_data))
        return data
    
    def apply_filter(self, data, filter):
        map_filter_result = True
        for map_filter in filter.get('MAP', []):
            map_filter_result = False
            if data['boro_code'] == map_filter:
                map_filter_result = True
                break

        scatter_plot_filter_result = True
        for sc_filter in filter.get('SCATTER_PLOT', []):
            scatter_plot_filter_result = False
            if data['review rate number'] == sc_filter:
                scatter_plot_filter_result = True
                break

        # Apply 'OR' filter on 'BAR_GRAPH' attribute
        bar_graph_filter_result = True
        for bar_graph_filter in filter.get('BAR_GRAPH', []):
            bar_graph_filter_result = False
            if all((data['year_range'] == bar_graph_filter['key'], data['room type'] == self.map_room_type[bar_graph_filter['roomType']])):
                bar_graph_filter_result = True
                break

        sunburst_filter_result = True
        for sunburst_filter in filter.get('SUNBURST', []):
            sunburst_filter_result = False
            if (data['price_range'] == sunburst_filter['0']) \
                  and (not sunburst_filter['1'] or data['instant_bookable'] == sunburst_filter['1']) \
                    and (not sunburst_filter['2'] or data['cancellation_policy'] == sunburst_filter['2']):
                
                sunburst_filter_result = True
                break

    # Apply 'AND' filter between all attributes
        return all((map_filter_result, scatter_plot_filter_result, bar_graph_filter_result, sunburst_filter_result))


    def get_mds_data(self):
        df = pd.DataFrame(
            MDS(dissimilarity='precomputed', random_state=49).fit_transform(1-self.data_set[NUMERICAL_COLUMNS].corr()), columns=["x", "y"]
        )
        return self._convert_mds_to_dict(json.loads(df.to_json(orient="records")))
    

    def _convert_mds_to_dict(self, mds_output):
        nodes = []
        for i in range(1, len(mds_output)):
            dist = ((mds_output[i]['x'] - mds_output[0]['x'])**2 + (mds_output[i]['y'] - mds_output[0]['y'])**2)**0.5
            nodes.append({'id': f'{NUMERICAL_COLUMNS[i]}', 'size': dist})

        max_dist = max(nodes, key=lambda x: x['size'])['size']
        min_dist = min(nodes, key=lambda x: x['size'])['size']

        for node in nodes:
            node['size'] = ((node['size'] - min_dist)/(max_dist-min_dist))*10 + 5

        
        links = []
        for i in range(1, len(mds_output)):
            for j in range(i+1, len(mds_output)):
                # print(mds_output[i])
                dist = ((mds_output[i]['x'] - mds_output[j]['x'])**2 + (mds_output[i]['y'] - mds_output[j]['y'])**2)**0.5
                links.append({'source': f'{NUMERICAL_COLUMNS[i]}', 'target': f'{NUMERICAL_COLUMNS[j]}', 'value': dist})

        links.sort(key=lambda x:x['value'], reverse=True)

        return {'nodes': nodes, 'links': links[:10]}


if __name__ == '__main__':
    svc = Service()
    svc.init_data()
    f = open('data.json', 'w')
    print(json.dumps(svc.get_mds_data()))
    f.write(svc.data_set.to_json(orient="records"))
    f.close()