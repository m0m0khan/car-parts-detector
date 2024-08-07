import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load an image and preprocess it for prediction with a trained model.

    Parameters:
    - img_path (str): The path to the image file.
    - target_size (tuple): The target size to resize the image.

    Returns:
    - preprocessed_img (numpy array): The preprocessed image.
    """
    try:
        # Load the image from the path
        img = image.load_img(img_path, target_size=target_size)
    except Exception as e:
        raise FileNotFoundError(f"Error loading image {img_path}: {e}")

    try:
        # Convert the image to an array
        img_array = image.img_to_array(img)

        # Expand dimensions to match the input shape of the model
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image for MobileNet (scaling pixel values)
        preprocessed_img = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    except Exception as e:
        raise ValueError(f"Error preprocessing image {img_path}: {e}")

    return preprocessed_img

def predict_car_components(img_path, model_path='car_hood_backdoor_detector.keras'):
    """
    Predict the probability scores for car components in an image using a trained model.

    Parameters:
    - img_path (str): The path to the image file to be predicted.
    - model_path (str): The path to the saved model file.

    Returns:
    - predictions (dict): A dictionary with the probability scores for the hood and left backdoor.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")

    try:
        # Load the trained model
        model = load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")

    try:
        # Load and preprocess the image
        preprocessed_img = load_and_preprocess_image(img_path)

        # Make predictions
        prediction = model.predict(preprocessed_img)

        # Extract probabilities
        hood_probability = prediction[0][0]
        backdoor_probability = prediction[0][1]
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

    # Return the predictions as a dictionary
    return {
        "hood_probability": float(hood_probability),
        "backdoor_probability": float(backdoor_probability)
    }

