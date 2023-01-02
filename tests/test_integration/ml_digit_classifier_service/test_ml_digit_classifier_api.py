from PIL import Image
import pytest
import requests
import base64
import json

MODEL_NAME = 'digit-classifier'
HOST = 'localhost'
PORT = '8080'
LIVENESS_ENDPOINT = '/'
MODEL_LIVENESS_ENDPOINT = f'/v1/models/{MODEL_NAME}'
MODEL_INFERENCE_ENDPOINT = f'/v1/models/{MODEL_NAME}:predict'

def test_ml_service_liveness():
    
    # test service liveness
    service_liveness_url = f'http://{HOST}:{PORT}{LIVENESS_ENDPOINT}'
    
    response = requests.get(service_liveness_url)
    
    assert response.json() == {"status": "alive"}
    assert response.status_code == 200
    
def test_ml_model_liveness():
    
    # test model liveness
    model_liveness_url = f'http://{HOST}:{PORT}{MODEL_LIVENESS_ENDPOINT}'
    
    response = requests.get(model_liveness_url)
    
    assert response.json() == {"name": MODEL_NAME, "ready": True}
    assert response.status_code == 200

@pytest.mark.parametrize(
    'image_path,actual_digit',
    [
        ('/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/10000/1/1_1.jpg','1'), # classification failure
        ('/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/10000/2/2_1.jpg','2'), # classification failure
        ('/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/10000/3/3_1.jpg','3'),
        ('/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/10000/4/4_1.jpg','4'),
        ('/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/10000/5/5_1.jpg','5'), # classification failure
        ('/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/10000/6/6_1.jpg','6'),
        ('/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/10000/7/7_1.jpg','7'), # classification failure
        ('/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/10000/8/8_1.jpg','8'), # classification failure
        ('/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/10000/9/9_1.jpg','9'), # classification failure
    ]
)
def test_ml_service_inference(image_path,actual_digit):
    
    model_inference_url = f'http://{HOST}:{PORT}{MODEL_INFERENCE_ENDPOINT}'
        
    with open(image_path, 'rb') as open_file:
        im_bytes = open_file.read()
               
    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    
    payload = json.dumps({"instances": [im_b64]})
    response = requests.post(model_inference_url, data=payload, headers=headers)
    predicted_digit = response.json()['predictions'][0]
    
    assert response.status_code == 200
    assert predicted_digit == actual_digit