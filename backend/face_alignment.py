import numpy as np
import pandas as pd 
import math 
import cv2
from PIL import Image

resp = {
    "face_1": {
        "score": 0.9993440508842468,
        "facial_area": [155, 81, 434, 443],
        "landmarks": {
            "right_eye": [257.82974, 209.64787],
            "left_eye": [374.93427, 251.78687],
            "nose": [303.4773, 299.91144],
            "mouth_right": [228.37329, 338.73193],
            "mouth_left": [320.21982, 374.58798],
        },
    }
}

x_1,y_1 = resp['face_1']['landmarks']['right_eye']
x_2, y_2 = resp['face_1']['landmarks']['left_eye']
x_1, y_1, x_2, y_2
a = abs(y_1 - y_2)
b = abs(x_2 - x_1)
c = math.sqrt((a*a) + (b * b)) 
a, b, c

cos_alpha = (b * b + c * c - a*a) / (2 * b * c)
cos_alpha

alpha = np.arccos(cos_alpha)
alpha

alpha = (alpha * 180) / math.pi 
alpha

new_image = image.fromarray()