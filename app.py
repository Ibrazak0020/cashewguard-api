from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import os

app = Flask(__name__)
CORS(app)

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cashew_model_final.tflite')

# Class names in exact order from your training
CLASS_NAMES = ['anthracnose', 'gumosis', 'healthy', 'leaf_miner', 'red_rust']

# Display names shown in the app
DISPLAY_NAMES = {
    'anthracnose': 'Anthracnose',
    'gumosis':     'Gumosis',
    'healthy':     'Healthy',
    'leaf_miner':  'Leaf Miner',
    'red_rust':    'Red Rust',
}

# ============================================
# LOAD TFLITE MODEL
# ============================================
interpreter = None

def load_model():
    global interpreter
    try:
        print(f'🔍 Model path: {MODEL_PATH}')
        print(f'🔍 Model exists: {os.path.exists(MODEL_PATH)}')
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print('✅ Model loaded successfully')
        # Print input details to confirm
        input_details = interpreter.get_input_details()
        print(f'✅ Input shape: {input_details[0]["shape"]}')
        return True
    except Exception as e:
        print(f'❌ Error loading model: {e}')
        return False

# ============================================
# IMAGE PREPROCESSING
# ============================================
def preprocess_image(image_data):
    try:
        # Remove base64 header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB
        image = image.convert('RGB')

        # Resize to 224x224
        image = image.resize((224, 224))

        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f'❌ Error preprocessing image: {e}')
        return None

# ============================================
# RUN PREDICTION
# ============================================
def run_prediction(img_array):
    try:
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        return output[0]

    except Exception as e:
        print(f'❌ Error running prediction: {e}')
        return None

# ============================================
# GET SEVERITY LEVEL
# ============================================
def get_severity(disease_key, infected_area):
    if disease_key == 'healthy':
        return 'Healthy'
    if infected_area > 60:
        return 'Severe'
    elif infected_area > 30:
        return 'Moderate'
    elif infected_area > 10:
        return 'Mild'
    else:
        return 'Mild'

# ============================================
# ROUTES
# ============================================

# Home route — test if API is running
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status':  'CashewGuard AI API is running ✅',
        'model':   'ResNet TFLite',
        'classes': CLASS_NAMES,
        'version': '1.0.0'
    })

# Health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'healthy',
        'model_loaded': interpreter is not None
    })

# Main prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Preprocess
        img_array = preprocess_image(data['image'])
        if img_array is None:
            return jsonify({'error': 'Failed to process image'}), 400

        # Predict
        predictions = run_prediction(img_array)
        if predictions is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Get top result
        predicted_index = int(np.argmax(predictions))
        confidence      = float(predictions[predicted_index])
        disease_key     = CLASS_NAMES[predicted_index]
        disease_name    = DISPLAY_NAMES[disease_key]

        # Calculate infected area
        if disease_key == 'healthy':
            infected_area = 0.0
        else:
            infected_area = round(confidence * 80, 1)

        # Get severity
        severity = get_severity(disease_key, infected_area)

        # All class probabilities
        all_predictions = {
            DISPLAY_NAMES[CLASS_NAMES[i]]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        print(f'✅ Result: {disease_name} | Confidence: {confidence*100:.1f}% | Severity: {severity}')

        return jsonify({
            'success':         True,
            'disease':         disease_name,
            'disease_key':     disease_key,
            'confidence':      round(confidence, 4),
            'severity':        severity,
            'infected_area':   infected_area,
            'all_predictions': all_predictions,
        })

    except Exception as e:
        print(f'❌ Error in predict: {e}')
        return jsonify({'error': str(e)}), 500

# ============================================
# START SERVER
# ============================================
if __name__ == '__main__':
    print('🌱 Starting CashewGuard AI API...')
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
else:
    # Called by gunicorn on Render
    load_model()
