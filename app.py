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

MODEL_PATH     = os.path.join(os.path.dirname(__file__), 'cashew_model_final.tflite')
VALIDATOR_PATH = os.path.join(os.path.dirname(__file__), 'leaf_validator.tflite')

CLASS_NAMES = ['anthracnose', 'gumosis', 'healthy', 'leaf_miner', 'red_rust']

DISPLAY_NAMES = {
    'anthracnose': 'Anthracnose',
    'gumosis':     'Gumosis',
    'healthy':     'Healthy',
    'leaf_miner':  'Leaf Miner',
    'red_rust':    'Red Rust',
}

# ✅ Optimal threshold from binary classifier training (98.12% accuracy)
VALIDATOR_THRESHOLD = 0.3

# ============================================
# LOAD DISEASE MODEL
# ============================================
interpreter = None

def load_model():
    global interpreter
    try:
        print(f'🔍 Model path: {MODEL_PATH}')
        print(f'🔍 Model exists: {os.path.exists(MODEL_PATH)}')
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print('✅ Disease model loaded successfully')
        input_details = interpreter.get_input_details()
        print(f'✅ Input shape: {input_details[0]["shape"]}')
        return True
    except Exception as e:
        print(f'❌ Error loading disease model: {e}')
        return False

# ============================================
# ✅ LOAD LEAF VALIDATOR MODEL
# ============================================
validator_interpreter = None

def load_validator():
    global validator_interpreter
    try:
        print(f'🔍 Validator path: {VALIDATOR_PATH}')
        print(f'🔍 Validator exists: {os.path.exists(VALIDATOR_PATH)}')
        validator_interpreter = tf.lite.Interpreter(model_path=VALIDATOR_PATH)
        validator_interpreter.allocate_tensors()
        print('✅ Leaf validator model loaded successfully')
        
        # Print validator input/output details for debugging
        input_details = validator_interpreter.get_input_details()
        output_details = validator_interpreter.get_output_details()
        print(f'✅ Validator input shape: {input_details[0]["shape"]}')
        print(f'✅ Validator output shape: {output_details[0]["shape"]}')
        
        return True
    except Exception as e:
        print(f'❌ Error loading validator model: {e}')
        return False

# ============================================
# IMAGE PREPROCESSING
# ============================================
def preprocess_image(image_data):
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        image = image.resize((224, 224))

        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, image

    except Exception as e:
        print(f'❌ Error preprocessing image: {e}')
        return None, None

# ============================================
# ✅ RUN LEAF VALIDATOR — binary classifier
# FIXED: Now correctly interprets model output
# ============================================
def run_validator(img_array):
    try:
        if validator_interpreter is None:
            print('⚠️ Validator not loaded — skipping')
            return None

        input_details  = validator_interpreter.get_input_details()
        output_details = validator_interpreter.get_output_details()

        validator_interpreter.set_tensor(input_details[0]['index'], img_array)
        validator_interpreter.invoke()

        output = validator_interpreter.get_tensor(output_details[0]['index'])
        raw_value = float(output[0][0])
        
        # 🔍 Debug output
        print(f'🔍 Validator raw output: {raw_value:.4f}')
        
        # 🔧 FIX: Invert the interpretation
        # Model was trained with: 0 = cashew leaf, 1 = non-cashew leaf
        # So lower confidence = more likely to be cashew leaf
        # Convert to a "cashew leaf confidence" score (0-1, higher = more likely leaf)
        cashew_confidence = 1.0 - raw_value
        
        print(f'🌿 Converted cashew confidence: {cashew_confidence:.4f}')
        
        return cashew_confidence

    except Exception as e:
        print(f'❌ Validator error: {e}')
        return None

# ============================================
# RUN DISEASE PREDICTION
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
# INFECTED AREA CALCULATION
# ============================================
def get_infected_area(disease_key, confidence):
    if disease_key == 'healthy':
        return 0.0
    infected = round(max(1.0, confidence * 100), 1)
    return min(infected, 100.0)

# ============================================
# SEVERITY LEVEL
# ============================================
def get_severity(disease_key, infected_area):
    if disease_key == 'healthy':
        return 'Healthy'
    if infected_area <= 25:
        return 'Mild'
    elif infected_area <= 50:
        return 'Moderate'
    else:
        return 'Severe'

# ============================================
# ROUTES
# ============================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status':           'CashewGuard AI API is running ✅',
        'disease_model':    'Best CNN Variant (Tuned)',
        'validator_model':  'MobileNetV2 Binary Classifier',
        'classes':          CLASS_NAMES,
        'version':          '3.0.1'  # Updated version
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':             'healthy',
        'model_loaded':       interpreter is not None,
        'validator_loaded':   validator_interpreter is not None,
        'validator_threshold': VALIDATOR_THRESHOLD,
    })

# ============================================
# ✅ FIXED: VALIDATE ENDPOINT
# Now correctly interprets validator output
# ============================================
@app.route('/validate', methods=['POST'])
def validate():
    """
    Validates if image is a cashew leaf using leaf_validator.tflite.
    Returns: { 'is_leaf': bool, 'confidence': float, 'message': str }
    """
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Preprocess image
        img_array, pil_image = preprocess_image(data['image'])
        if img_array is None:
            return jsonify({'error': 'Failed to process image'}), 400

        # ✅ Run leaf_validator.tflite (now returns cashew confidence)
        cashew_confidence = run_validator(img_array)

        if cashew_confidence is None:
            # Validator not loaded — fall back to color analysis
            print('⚠️ Validator unavailable — using color analysis fallback')
            is_leaf, reason = _color_analysis(pil_image)
            message = 'This does not appear to be a cashew leaf. Please upload a clear photo of a cashew leaf.' \
                      if not is_leaf else 'Image looks good. Proceeding to analysis.'
            return jsonify({
                'is_leaf':    is_leaf,
                'confidence': 0.0,
                'reason':     reason,
                'message':    message,
            })

        # 🔧 FIXED: Now cashew_confidence is already inverted (higher = more likely leaf)
        is_cashew = cashew_confidence >= VALIDATOR_THRESHOLD

        print(f'🌿 Validator confidence (cashew): {cashew_confidence*100:.1f}% | '
              f'threshold: {VALIDATOR_THRESHOLD} | isCashew: {is_cashew}')

        if not is_cashew:
            return jsonify({
                'is_leaf':    False,
                'confidence': round(cashew_confidence, 4),
                'reason':     'not_cashew_leaf',
                'message':    'This does not appear to be a cashew leaf. '
                              'Please upload a clear photo of a cashew leaf.',
            })

        return jsonify({
            'is_leaf':    True,
            'confidence': round(cashew_confidence, 4),
            'reason':     'valid',
            'message':    'Image validated as cashew leaf. Proceeding to analysis.',
        })

    except Exception as e:
        print(f'❌ Validation error: {e}')
        return jsonify({'error': str(e)}), 500


def _color_analysis(pil_image):
    """Fallback color analysis if validator model is unavailable."""
    img_array       = np.array(pil_image.resize((64, 64)))
    r               = img_array[:, :, 0].astype(float)
    g               = img_array[:, :, 1].astype(float)
    b               = img_array[:, :, 2].astype(float)
    mean_r          = np.mean(r)
    mean_g          = np.mean(g)
    mean_b          = np.mean(b)
    green_dominant  = (mean_g > mean_r) and (mean_g > mean_b)
    green_pixels    = np.sum((g > r * 0.85) & (g > b * 0.85) & (g > 60))
    total_pixels    = img_array.shape[0] * img_array.shape[1]
    green_ratio     = green_pixels / total_pixels
    mean_brightness = (mean_r + mean_g + mean_b) / 3

    if mean_brightness < 30:
        return False, 'too_dark'
    if mean_brightness > 230:
        return False, 'too_bright'
    if not green_dominant and green_ratio < 0.15:
        return False, 'not_green'
    return True, 'valid'


# ============================================
# ✅ FIXED: PREDICT ENDPOINT
# Now correctly interprets validator output
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Step 1: Preprocess
        img_array, pil_image = preprocess_image(data['image'])
        if img_array is None:
            return jsonify({'error': 'Failed to process image'}), 400

        # Step 2: Run disease prediction
        predictions = run_prediction(img_array)
        if predictions is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Step 3: Validate with binary classifier
        cashew_confidence = run_validator(img_array)

        if cashew_confidence is not None:
            # 🔧 FIXED: cashew_confidence is already inverted (higher = more likely leaf)
            is_leaf = cashew_confidence >= VALIDATOR_THRESHOLD
            print(f'🌿 Predict-level validator: {cashew_confidence*100:.1f}% | isCashew: {is_leaf}')
        else:
            # Fallback to color analysis
            is_leaf, _ = _color_analysis(pil_image)

        if not is_leaf:
            return jsonify({
                'success':         False,
                'disease':         'Unrecognized',
                'disease_key':     'unrecognized',
                'confidence':      0.0,
                'severity':        'Unknown',
                'infected_area':   0.0,
                'all_predictions': {},
                'reason':          'not_cashew_leaf',
                'message':         'The uploaded image does not appear to be a cashew leaf.',
            })

        # Step 4: Process valid prediction
        predicted_index = int(np.argmax(predictions))
        confidence      = float(predictions[predicted_index])
        disease_key     = CLASS_NAMES[predicted_index]
        disease_name    = DISPLAY_NAMES[disease_key]

        infected_area = get_infected_area(disease_key, confidence)
        severity      = get_severity(disease_key, infected_area)

        all_predictions = {
            DISPLAY_NAMES[CLASS_NAMES[i]]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        print(f'✅ Result: {disease_name} | Confidence: {confidence*100:.1f}% | '
              f'Infected Area: {infected_area}% | Severity: {severity}')

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
# ✅ NEW: DEBUG ENDPOINT to test validator
# ============================================
@app.route('/debug/validator', methods=['POST'])
def debug_validator():
    """
    Debug endpoint to test validator raw output
    Returns raw model output without any interpretation
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img_array, pil_image = preprocess_image(data['image'])
        if img_array is None:
            return jsonify({'error': 'Failed to process image'}), 400

        if validator_interpreter is None:
            return jsonify({'error': 'Validator not loaded'}), 500

        input_details = validator_interpreter.get_input_details()
        output_details = validator_interpreter.get_output_details()

        validator_interpreter.set_tensor(input_details[0]['index'], img_array)
        validator_interpreter.invoke()

        raw_output = validator_interpreter.get_tensor(output_details[0]['index'])
        raw_value = float(raw_output[0][0])
        
        # Also run color analysis for comparison
        is_leaf_color, color_reason = _color_analysis(pil_image)

        return jsonify({
            'raw_output': raw_value,
            'cashew_confidence': 1.0 - raw_value,  # Inverted
            'threshold': VALIDATOR_THRESHOLD,
            'is_cashew_by_model': (1.0 - raw_value) >= VALIDATOR_THRESHOLD,
            'color_analysis': {
                'is_leaf': is_leaf_color,
                'reason': color_reason
            },
            'interpretation': f"Model output {raw_value:.4f} means {'cashew leaf' if raw_value < 0.5 else 'non-cashew leaf'} (approximately)"
        })

    except Exception as e:
        print(f'❌ Debug error: {e}')
        return jsonify({'error': str(e)}), 500

# ============================================
# START SERVER
# ============================================
if __name__ == '__main__':
    print('🌱 Starting CashewGuard AI API v3.0.1 (Fixed Validator)...')
    print('=' * 50)
    load_model()
    load_validator()
    print('=' * 50)
    print('🚀 Server ready!')
    print('📌 Use /debug/validator to test validator output')
    print('📌 Use /validate to check if image is a cashew leaf')
    print('📌 Use /predict for full disease prediction')
    print('=' * 50)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
else:
    load_model()
    load_validator()
