from flask import Flask, request
from flask import jsonify
from flask_cors import CORS
from sklearn.manifold import TSNE
import numpy
app = Flask(__name__)
cors = CORS(app)


@app.route('/make-data', methods=['POST'])
def post():
    data = request.json
    print(data)
    matrix = list(map(lambda d: d['scores'], data))
    print(matrix)
    tsne = TSNE(n_components=2).fit_transform(matrix)
    result = numpy.around(tsne, decimals=3)
    print(result)
    return jsonify(result.tolist())


if __name__ == '__main__':
    app.run(debug=True)
