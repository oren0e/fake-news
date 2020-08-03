from flask import Flask, request

from backend.ft_model import NNClassifier

PORT: int = 9000

app = Flask(__name__)

@app.route('/fake_news_prediction', methods=['POST'])
def make_prediction() -> str:
    data: str = request.get_data().decode('utf-8')

    assert type(data) == str
    model = NNClassifier(data)

    return model.predict()



if __name__ == '__main__':
    app.run(port=PORT, debug=False)