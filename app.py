from flask import Flask
from service import Service

server = Flask(__name__)
server.config['JSON_SORT_KEYS'] = False

m = Service()

@server.route('/api/data')
def get_data():
    return m.data_set.to_json(orient="records")


server.run(port=5668, debug=True)