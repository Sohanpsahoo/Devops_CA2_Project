from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import io
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# Keep fallback for testing but recommend using .env
genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'AIzaSyAK8obivhQ8T9mF-dGC-SnaZ7GutGQF4e0'))

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Class names
CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
               'River', 'SeaLake']

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize image to 64x64 (EuroSAT dataset size)
    image = image.resize((64, 64))
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Open and preprocess the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'type': predicted_class,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/insights', methods=['POST'])
def insights():
    data = request.json
    land_type = data.get('landType')
    if not land_type:
        return jsonify({'error': 'No landType provided'}), 400
        
    try:
        gemini_model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={"response_mime_type": "application/json"}
        )
        prompt = f"""
        Analyze the {land_type} land cover type. You MUST return a valid JSON object containing exactly the following 4 keys, with the content formatted using markdown:
        1. "futureTrends": Analyze future trends (climate change, urbanization, environmental factors) in 100-200 words.
        2. "urbanPlanning": Provide urban planning insights (sustainable development, infrastructure) in 100-200 words.
        3. "conservation": Suggest conservation strategies (biodiversity protection, sustainable management) in 100-200 words.
        4. "recommendations": Generate 4-5 specific, actionable recommendations (1-2 sentences each).
        """
        
        response = gemini_model.generate_content(prompt)
        import json
        results = json.loads(response.text)
            
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)