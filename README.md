# Land Classification Web App

A minimalist web application for classifying land types in satellite/aerial images using deep learning.

## Features

- Upload satellite/aerial images
- Automatic image preprocessing
- Real-time classification using a pre-trained CNN model
- Simple and intuitive user interface
- Shows classification results with confidence scores

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (this will download the EuroSAT dataset and train the model):
```bash
python train_model.py
```

3. Start the Flask server:
```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click the "Choose Image" button to select a satellite/aerial image
2. The selected image will be displayed in the preview area
3. Click the "Classify" button to process the image
4. The results will show the predicted land type and confidence score

## Model Details

- Uses a Convolutional Neural Network (CNN) trained on the EuroSAT dataset
- Input image size: 64x64 pixels
- Output classes: 10 different land types including agricultural, urban, water, forest, etc.
- Model architecture: 3 convolutional layers with max pooling, followed by dense layers

## Technical Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- Machine Learning: TensorFlow/Keras
- Image Processing: Pillow

## Note

The first run will download the EuroSAT dataset (approximately 2GB) and train the model, which may take some time depending on your system's specifications. 