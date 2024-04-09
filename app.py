import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow INFO messages
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model('rice_leaf_disease_model.h5')

# Function to preprocess the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Function to predict the class of the input image
def predict_image_class(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    class_index = np.argmax(prediction)
    return class_index

# Route for predicting rice leaf disease class
@app.route('/predict', methods=['POST'])
def predict():
    # Check if image file is present in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    # Save the image to a temporary file
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)

    # Predict the class of the input image
    predicted_class_index = predict_image_class(temp_image_path)

    # Map class index to class label
    class_labels = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']
    predicted_class_label = class_labels[predicted_class_index]

    return jsonify({'predicted_class': predicted_class_label}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
