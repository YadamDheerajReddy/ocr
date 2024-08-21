from tesseract.ocr_tesseract import ocr_using_tesseract
from model.cnn_model import create_cnn_model, train_model, predict_text
from preprocessing.preprocess_image import preprocess_image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, y_train, X_test, y_test

def main(image_path):
    # Use Tesseract OCR for general text extraction
    text = ocr_using_tesseract(image_path)
    print(f"Recognized Text using Tesseract: {text}")
    
    # Save recognized text to a TXT file
    with open("recognized_text.txt", "w") as file:
        file.write(text)
    
    # Alternatively, use the CNN model for digit recognition (example)
    model = create_cnn_model()
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    train_model(model, X_train, y_train, X_test, y_test)
    
    # Preprocess and predict for a given image
    image = preprocess_image(image_path)
    digit = predict_text(model, image)
    print(f"Recognized Digit using CNN: {digit}")

if __name__ == "__main__":
    main('data/sample_image.png')
