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

CLASS_NAMES = ['anthracnose', 'gumosis', 'healthy', 'leaf_miner', 'red_rust']

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
# LEAF VALIDATION
# ============================================
def is_cashew_leaf(pil_image, predictions):
    """
    Returns (is_leaf: bool, reason: str)
    Validates using color analysis + model confidence threshold.
    """

    # Signal 1: Model confidence threshold
    max_confidence = float(np.max(predictions))
    if max_confidence < 0.40:
        print(f'❌ Low confidence: {max_confidence:.2%} — not a cashew leaf')
        return False, 'low_confidence'

    # Signal 2: Green dominance color check
    img_array = np.array(pil_image.resize((64, 64)))

    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)

    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    print(f'🎨 RGB means — R:{mean_r:.1f} G:{mean_g:.1f} B:{mean_b:.1f}')

    green_dominant = (mean_g > mean_r) and (mean_g > mean_b)
    green_pixels   = np.sum((g > r * 0.85) & (g > b * 0.85) & (g > 60))
    total_pixels   = img_array.shape[0] * img_array.shape[1]
    green_ratio    = green_pixels / total_pixels

    print(f'🌿 Green dominant: {green_dominant} | Green ratio: {green_ratio:.2%}')

    if not green_dominant and green_ratio < 0.15:
        print('❌ Not enough green — not a cashew leaf')
        return False, 'not_green'

    # Signal 3: Brightness check
    mean_brightness = (mean_r + mean_g + mean_b) / 3
    if mean_brightness < 30:
        print(f'❌ Image too dark: brightness={mean_brightness:.1f}')
        return False, 'too_dark'
    if mean_brightness > 230:
        print(f'❌ Image too bright/blank: brightness={mean_brightness:.1f}')
        return False, 'too_bright'

    print('✅ Image passed leaf validation')
    return True, 'valid'

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
# ✅ UPDATED: INFECTED AREA CALCULATION
# ============================================
def get_infected_area(disease_key, confidence):
    """
    Estimates infected leaf area percentage.
      - Healthy class always returns 0%
      - Disease classes: confidence scaled to infected area (1–100%)
    """
    if disease_key == 'healthy':
        return 0.0
    infected = round(max(1.0, confidence * 100), 1)
    return min(infected, 100.0)

# ============================================
# ✅ UPDATED: SEVERITY LEVEL
# Standard plant pathology severity scale:
#   Healthy  → predicted class is healthy (0% infected)
#   Mild     → 1%  to 25% infected area
#   Moderate → 26% to 50% infected area
#   Severe   → above 50% infected area
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
        'status':  'CashewGuard AI API is running ✅',
        'model':   'Best CNN Variant (Tuned)',
        'classes': CLASS_NAMES,
        'version': '2.0.0'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'healthy',
        'model_loaded': interpreter is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Step 1: Preprocess image
        img_array, pil_image = preprocess_image(data['image'])
        if img_array is None:
            return jsonify({'error': 'Failed to process image'}), 400

        # Step 2: Run prediction
        predictions = run_prediction(img_array)
        if predictions is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Step 3: Validate — is this actually a cashew leaf?
        is_leaf, reason = is_cashew_leaf(pil_image, predictions)

        if not is_leaf:
            print(f'🚫 Image rejected: {reason}')
            return jsonify({
                'success':         False,
                'disease':         'Unrecognized',
                'disease_key':     'unrecognized',
                'confidence':      0.0,
                'severity':        'Unknown',
                'infected_area':   0.0,
                'all_predictions': {},
                'reason':          reason,
                'message':         'The uploaded image does not appear to be a cashew leaf. Please upload a clear photo of a cashew leaf.',
            })

        # Step 4: Process valid prediction
        predicted_index = int(np.argmax(predictions))
        confidence      = float(predictions[predicted_index])
        disease_key     = CLASS_NAMES[predicted_index]
        disease_name    = DISPLAY_NAMES[disease_key]

        # ✅ UPDATED: Use new infected area + severity calculations
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
# START SERVER
# ============================================
if __name__ == '__main__':
    print('🌱 Starting CashewGuard AI API v2.0...')
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
else:
    load_model()
