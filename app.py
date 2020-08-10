from flask import Flask, request, jsonify

from backend.ft_model import NNClassifier

app = Flask(__name__, static_folder='./build', static_url_path='/')

@app.before_first_request
def _init_classifier() -> None:
    global model
    model = NNClassifier()
    model.initial_load()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction() -> dict:
    data: str = request.get_data().decode('utf-8')

    assert type(data) == str
    model.get_data(data)
    output = model.predict()

    return jsonify(result=output)


if __name__ == '__main__':
    app.run(debug=False)