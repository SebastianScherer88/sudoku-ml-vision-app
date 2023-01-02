import base64
import io
from typing import List

import PIL
from pydantic import BaseModel


class DigitImage(BaseModel):
    
    instances: List[str] # base64 encoded image file byte strings

def decode_image_file_from_http(image_file_data: str):

    # undo base64 encoding step
    im_b = base64.b64decode(image_file_data)

    image_byte_stream = io.BytesIO(im_b)
    image = PIL.Image.open(image_byte_stream)
        
    return image