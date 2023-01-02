from PIL import Image
import pytest
import requests
import base64
import json
import logging

logger = logging.getLogger('parse-image-service-integration-test')
logging.basicConfig()
# log_level = logging.DEBUG
# logger.setLevel(log_level)

HOST = 'localhost'
PORT = '8000'
LIVENESS_ENDPOINT = '/'
IMAGE_PARSE_ENDPOINT = f'/parse_image'

def test_parse_service_liveness():
    
    # test service liveness
    service_liveness_url = f'http://{HOST}:{PORT}{LIVENESS_ENDPOINT}'
    
    response = requests.get(service_liveness_url)
    
    assert response.json() == {"status": "alive"}
    assert response.status_code == 200
    
@pytest.mark.parametrize(
    'image_path',
    [
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-1.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-2.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-3.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-4.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-5.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-6.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-7.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-8.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-9.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-10.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-11.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-12.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-13.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-14.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-15.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-16.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-17.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-18.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-19.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-20.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-21.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-22.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-23.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-24.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-25.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-26.PNG',
        '/home/ubuntu/repositories/sudoku-ml-vision-app/temporary/sudoku-grid-images/manual-extracts/grid-27.PNG',
    ]
)
def test_parse_service_inference(image_path):
    
    model_inference_url = f'http://{HOST}:{PORT}{IMAGE_PARSE_ENDPOINT}'
        
    with open(image_path, 'rb') as open_file:
        im_bytes = open_file.read()
               
    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    
    payload = json.dumps({"instances": [im_b64]})
    response = requests.post(model_inference_url, data=payload, headers=headers)
    response_data = response.json()['instances'][0]
    
    assert response.status_code == 200
    
    for i in range(81):
        if response_data['image_parsed']:
            assert response_data['parsed_cells'][i] != ''
        elif not response_data['image_parsed']:
            assert response_data['parsed_cells'][i] == ''