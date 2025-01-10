import requests
import json

if __name__ == '__main__':
    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 1,
        'user_name': 'user',
        'password': 'password',
    }))
    print(response)

    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 369,
        'user_name': 'ca_user',
        'password': 'ca_user',
    }))
    print(response)

    # import random
    # import string
    #
    # length = 3
    # random_string = ''.join([random.choice(string.ascii_letters.lower() + string.digits) for _ in range(length)])
    # print(random_string)