import cv2

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh_image
