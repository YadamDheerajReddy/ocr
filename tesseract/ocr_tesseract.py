import pytesseract
from PIL import Image
from preprocessing.preprocess_image import preprocess_image

def ocr_using_tesseract(image_path):
    image = preprocess_image(image_path)
    pil_image = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_image)
    return text
