import numpy as np
import cv2 as cv
from PIL import Image

def enhancer(file):
    img = Image.open(file)
    img = img.convert('L')
    img_arr = np.array(img)

    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img_arr)
    cl = Image.fromarray(cl1)
    
    return cl

