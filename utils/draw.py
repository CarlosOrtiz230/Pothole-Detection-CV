import cv2

def draw_box(img, box, color=(0, 255, 0), label=None):
    """
    Draws a rectangle and optional label on the image.

    Args:
        img (np.ndarray): RGB image.
        box (tuple): (x, y, w, h)
        color (tuple): RGB color.
        label (str): Optional label text.
    """
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, label, (x, y - 5), font, 0.5, color, 1, cv2.LINE_AA)

    return img
