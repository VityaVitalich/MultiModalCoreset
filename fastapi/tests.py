import pytest
import requests
import numpy as np

def test_rgb_infer():
    rgb_path = './test_imgs/rgb.png'
    files = {'rgb': open(rgb_path, 'rb')}
    url = 'http://127.0.0.1:8886/rgb_inference/'
    response = requests.get(url, files=files)

    files['rgb'].close()

    assert response.status_code == 200

def test_rgb_semseg_infer():
    rgb_path = './test_imgs/rgb.png'
    semseg_path = './test_imgs/semseg.png'
    files = {'rgb': open(rgb_path, 'rb'), 'semseg': open(semseg_path, 'rb')}
    url = 'http://127.0.0.1:8886/rgb_semantic_inference/'
    response = requests.get(url, files=files)

    files['rgb'].close()
    files['semseg'].close()

    assert response.status_code == 200

def test_semseg_infer():
    semseg_path = './test_imgs/semseg.png'
    files = {'semseg': open(semseg_path, 'rb')}
    url = 'http://127.0.0.1:8886/semantic_inference/'
    response = requests.get(url, files=files)

    files['semseg'].close()

    assert response.status_code == 200


def test_correct_fine_tune():
    rgb_path = './test_imgs/rgb.png'
    depth_path = './test_imgs/depth.png'
    url = 'http://127.0.0.1:8886/fine_tune_rgb/'

    files = []
    for i in range(2):
        files.append(('rgb_ls', (f'rgb_image{i}.png', open(rgb_path, 'rb'), 'image/png')))
    for i in range(2):
        files.append(('depth_ls', (f'depth_image{i}.png', open(depth_path, 'rb'), 'image/png')))

    data = {'token': 'my_token'}

    # Make a POST request to the endpoint
    response = requests.post(url, files=files, data=data)

    for _, file_tuple in files:
        if isinstance(file_tuple, tuple):
            file_tuple[1].close()

    assert response.json()['message'] == "RGB Model successfully fine-tuned"

def test_different_length_fine_tune():
    rgb_path = './test_imgs/rgb.png'
    depth_path = './test_imgs/depth.png'
    url = 'http://127.0.0.1:8886/fine_tune_rgb/'

    files = []
    for i in range(3):
        files.append(('rgb_ls', (f'rgb_image{i}.png', open(rgb_path, 'rb'), 'image/png')))
    for i in range(2):
        files.append(('depth_ls', (f'depth_image{i}.png', open(depth_path, 'rb'), 'image/png')))

    data = {'token': 'my_token'}

    # Make a POST request to the endpoint
    response = requests.post(url, files=files, data=data)

    for _, file_tuple in files:
        if isinstance(file_tuple, tuple):
            file_tuple[1].close()

    assert response.json()['message'] == "All Images should have target"

def test_unauth_fine_tune():
    rgb_path = './test_imgs/rgb.png'
    depth_path = './test_imgs/depth.png'
    url = 'http://127.0.0.1:8886/fine_tune_rgb/'

    files = []
    for i in range(2):
        files.append(('rgb_ls', (f'rgb_image{i}.png', open(rgb_path, 'rb'), 'image/png')))
    for i in range(2):
        files.append(('depth_ls', (f'depth_image{i}.png', open(depth_path, 'rb'), 'image/png')))

    data = {'token': 'my_token_incorrect'}

    # Make a POST request to the endpoint
    response = requests.post(url, files=files, data=data)

    for _, file_tuple in files:
        if isinstance(file_tuple, tuple):
            file_tuple[1].close()

    assert response.json()['message'] == "Fine tuning is not permitted"

def test_too_many_img_fine_tune():
    rgb_path = './test_imgs/rgb.png'
    depth_path = './test_imgs/depth.png'
    url = 'http://127.0.0.1:8886/fine_tune_rgb/'

    files = []
    for i in range(10):
        files.append(('rgb_ls', (f'rgb_image{i}.png', open(rgb_path, 'rb'), 'image/png')))
    for i in range(10):
        files.append(('depth_ls', (f'depth_image{i}.png', open(depth_path, 'rb'), 'image/png')))

    data = {'token': 'my_token'}

    # Make a POST request to the endpoint
    response = requests.post(url, files=files, data=data)

    for _, file_tuple in files:
        if isinstance(file_tuple, tuple):
            file_tuple[1].close()

    assert response.json()['message'] == "Too much samples for a small service"


def test_embedder():
    rgb_path = './test_imgs/rgb.png'
    url = 'http://127.0.0.1:8886/rgb_embeddings/'

    files = {'rgb': open(rgb_path, 'rb')}


    # Make a POST request to the endpoint
    response = requests.get(url, files=files)

    files['rgb'].close()

    assert response.json()['message'] == "RGB embeddings successfully extracted"
    assert np.array(response.json()['embeddings']).shape[1] == 768

