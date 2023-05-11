from flask import Flask, request
from flask_cors import CORS
from service import Service

server = Flask(__name__)
# server.config['JSON_SORT_KEYS'] = False

CORS(server)
m = Service()

@server.route('/api/data', methods=['POST'])
def get_data():
    return m.get_filtered_data(request.json)


server.run(port=5668, debug=True)