import axios from 'axios';

export default (req, res) => {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json');

  if (req.method === 'POST' && req.body.text) {
    axios
      .post('http://localhost:9000/predict', {
        text: req.body.text,
      })
      .then(function (response) {
        res.end(JSON.stringify({ success: true, result: response.result }));
      })
      .catch(function (error) {
        res.end(JSON.stringify({ error: true, errorMessage: error.message }));
      });
  }
};
