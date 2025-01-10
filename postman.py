import requests
import json

if __name__ == '__main__':
    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 1,
        'user_name': 'user',
        'password': 'password',
    }))
    print(response)