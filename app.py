from flask import Flask, request, jsonify, send_from_directory

from backend.ft_model import NNClassifier

PORT: int = 9000

app = Flask(__name__, static_folder='./frontend')

@app.route('/')
def index_page():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def make_prediction() -> dict:
    data: str = request.get_data().decode('utf-8')

    assert type(data) == str
    model = NNClassifier(data)
    output = model.predict()

    return jsonify(result=output)



if __name__ == '__main__':
    app.run(port=PORT, debug=False)