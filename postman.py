import requests
import json

if __name__ == '__main__':
    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 1,
        'user_name': 'user_100',
        'password': 'user_100',
    }))
    print(response)

    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 2,
        'user_name': 'user_200',
        'password': 'user_200',
    }))
    print(response)
    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 3,
        'user_name': 'user_300',
        'password': 'user_300',
    }))
    print(response)
    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 4,
        'user_name': 'user_400',
        'password': 'user_400',
    }))
    print(response)

    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 5,
        'user_name': 'user_500',
        'password': 'user_500',
    }))

    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 6,
        'user_name': 'user_600',
        'password': 'user_600',
    }))
    print(response)

    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 7,
        'user_name': 'user_700',
        'password': 'user_700',
    }))
    print(response)

    print(response)
    response = requests.post('http://127.0.0.1:8000/register', data=json.dumps({
        'id': 369,
        'user_name': 'user_ca',
        'password': 'user_ca',
    }))



    # import random
    # import string
    #
    # length = 3
    # random_string = ''.join([random.choice(string.ascii_letters.lower() + string.digits) for _ in range(length)])
    # print(random_string)