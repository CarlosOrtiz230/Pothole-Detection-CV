import numpy as np

def draw_box(img: np.ndarray, box: tuple, color=(255, 0, 0)) -> np.ndarray:
    """
    Draws a rectangle on the image using NumPy.

    Args:
        img (np.ndarray): RGB image.
        box (tuple): (x, y, w, h)
        color (tuple): RGB color

    Returns:
        np.ndarray: Image with rectangle drawn.
    """
    x, y, w, h = box
    # Clamp coordinates
    h_img, w_img = img.shape[:2]
    x0, x1 = max(x, 0), min(x + w, w_img)
    y0, y1 = max(y, 0), min(y + h, h_img)

    # Top and bottom lines
    img[y0:y0+2, x0:x1] = color
    img[y1-2:y1, x0:x1] = color
    # Left and right lines
    img[y0:y1, x0:x0+2] = color
    img[y0:y1, x1-2:x1] = color

    return img
