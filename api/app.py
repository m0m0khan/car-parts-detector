from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import os
from werkzeug.utils import secure_filename
from inference import predict_car_components

app = Flask(__name__)
api = Api(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class Predict(Resource):
    def get(self):
        """
        Handle GET requests to check the API status.
        
        Returns:
        - JSON response indicating the API is running.
        """    
        return jsonify({"message": "Car component prediction API is running."})

    def post(self):
        """
        Handle POST requests to perform car component predictions on an uploaded image.
        
        Expects:
        - An image file uploaded via multipart/form-data with the key 'file'.
        
        Returns:
        - JSON response containing the prediction probabilities for car components.
        """
        # Check if the 'file' key is present in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']

        # If no file is selected        
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400
        
        # Save the file securely        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Perform prediction using the inference function
            model_path = 'car_hood_backdoor_detector.keras'
            predictions = predict_car_components(filepath, model_path=model_path)

            # Remove the uploaded file after prediction
            os.remove(filepath)
            
            # Check if predictions are serializable
            if isinstance(predictions, dict):
                try:
                    # Return predictions as JSON
                    return jsonify(predictions)
                except TypeError as e:
                    app.logger.error(f"JSON Serialization Error: {e}")
                    return jsonify({"error": "Error serializing predictions to JSON"}), 500
            else:
                return jsonify({"error": "Predictions are not in the expected format"}), 500
        
        except FileNotFoundError as e:
            app.logger.error(f"File Not Found Error: {e}")
            return jsonify({"error": f"File not found: {e}"}), 404
        except ValueError as e:
            app.logger.error(f"Value Error: {e}")
            return jsonify({"error": 
            f"Value error: {e}"}), 400
        except RuntimeError as e:
            app.logger.error(f"Runtime Error: {e}")
            return jsonify({"error": f"Runtime error: {e}"}), 500
        except Exception as e:
            app.logger.error(f"Unexpected Error: {e}")
            return jsonify({"error": "An unexpected error occurred"}), 500

# Add the Predict resource to the API
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

