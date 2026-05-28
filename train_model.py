from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
import requests
from tqdm import tqdm
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_dataset():
    # URL for EuroSAT dataset (updated URL)
    url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
    filename = "EuroSAT.zip"
    
    # Download the dataset
    print("Downloading dataset...")
    response = requests.get(url, stream=True, verify=False)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # Remove the zip file
    os.remove(filename)
    print("Dataset ready!")

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes in EuroSAT
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model():
    # Download dataset if not already present
    if not os.path.exists("2750"):  # EuroSAT images are in the '2750' directory
        print("Downloading dataset...")
        download_dataset()
    
    # Create data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        '2750',
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        '2750',
        target_size=(64, 64),
        batch_size=32,
        class_mode='sparse',
        subset='validation'
    )
    
    # Create and train model
    model = create_model()
    
    print("Training model...")
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10
    )
    
    # Save the model
    model.save('model.h5')
    print("Model saved as model.h5")

if __name__ == '__main__':
    train_model()