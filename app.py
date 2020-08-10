from flask import Flask, request, jsonify

from backend.ft_model import NNClassifier

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def make_prediction() -> dict:
    data: str = request.get_data().decode('utf-8')

    assert type(data) == str
    model = NNClassifier(data)
    output = model.predict()

    return jsonify(result=output)



if __name__ == '__main__':
    app.run(debug=True)