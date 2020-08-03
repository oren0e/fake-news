'''
You can read a news article from txt file and then store it in 'data'
'''

import requests

url = 'http://localhost:9000/fake_news_prediction'
data = 'the news are TRUE, we all must believe that!'
r = requests.post(url, data)

print(r.content.decode('utf-8'))