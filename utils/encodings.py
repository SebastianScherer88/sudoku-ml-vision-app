import base64
from typing import Union, Path, Literal
import io
import PIL
import cv2
import numpy as np

def encode_image_file_for_http(image_file_path: Union[Path,str]) -> str:
    '''
    Takes a file from disk and applies encoding convention to allow image data to be
    serialized to json.
    Useful for sending image data via HTTP.
    '''
    
    with open(image_file_path, 'rb') as open_file:
        im_bytes = open_file.read()
    
    # base64 encoding
    im_b64 = base64.b64encode(im_bytes)
    
    # utf decode so json can serialize it
    im_b64_str = im_b64.decode("utf8")
    
    return im_b64_str

def decode_image_file_from_http(image_file_data: str,
                                decode_target: Literal['pil','cv2']):

    # undo base64 encoding step
    im_b = base64.b64decode(image_file_data)

    if decode_target == 'pil':
        image_byte_stream = io.BytesIO(im_b)
        image = PIL.Image.open(image_byte_stream)
        
    elif decode_target == 'cv2':
        image_arr = np.frombuffer(im_b, np.uint8)
        image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR) # reads BGR color channel ordering (like the cv2.imread function - see above)
        
    return image