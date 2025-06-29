import numpy as np
from utils.preprocess import to_gray

def detect(img):

    #starter code that will work later
    
    """Detect potholes in the center region of the image.
    Returns:
        confidence (float): 0 to 1
        box (tuple): (x, y, width, height)
    """
    gray = to_gray(img)
    h, w = gray.shape

    # Define region of interest (center rectangle)
    x = w // 4
    y = h // 4
    w_roi = w // 2
    h_roi = h // 2

    roi = gray[y:y+h_roi, x:x+w_roi]
    mask = roi < 60  # threshold for dark pixels

    confidence = mask.mean()
    box = (x, y, w_roi, h_roi)

    return confidence, box
 