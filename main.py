from pymongo import MongoClient
import cv2
import base64
import numpy as np
from datetime import datetime
from PIL import Image
import os
from bson.binary import Binary
import io
from crud import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import base64
import io
import numpy as np
from PIL import Image

def load_images_from_db(dataset_type, target_size=(224, 224)):
    global client, db, collection
    images = []
    labels = []

    for doc in collection.find({'dataset_type': dataset_type}):
        image_data = doc['image']
        
        # Decode the base64 string if it's a string type
        if isinstance(image_data, str):
            if image_data.startswith('data:image/jpeg;base64,'):
                image_data = image_data.split(',')[1]
            image_data = base64.b64decode(image_data)
        
        # Open the image using PIL
        try:
            image = Image.open(io.BytesIO(image_data))
            # Resize the image to target size
            image = image.resize(target_size)
            images.append(np.array(image))
            labels.append(doc['class'])
        except Exception as e:
            print(f"Error loading image: {e}")

    client.close()
    
    return np.array(images), np.array(labels)



def upload_dataset(base_path, collection):
    datasets = ['seg_train', 'seg_test', 'seg_pred']
    
    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        if not os.path.exists(dataset_path):
            print(f"Dataset folder '{dataset}' does not exist. Skipping.")
            continue

        #get dataset type
        dataset_type = dataset.split('_')[1] if '_' in dataset else dataset
        
        #subfolders represent classes
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            #loop over images
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                
                try:
                    #encode image
                    with open(img_path, "rb") as img:
                        image_binary = base64.b64encode(img.read()).decode('utf-8')
                    
                    #metadata
                    image = Image.open(img_path)
                    metadata = {
                        "filename": img_file,
                        "class": class_name,
                        "dataset_type": dataset_type,  # Train, Test, or Pred
                        "dimensions": image.size,
                        "color_mode": image.mode
                    }
                    
                    #Upload into MongoDB
                    collection.insert_one({"image": image_binary, **metadata})
                    print(f"Uploaded {img_file} in class '{class_name}' ({dataset_type}).")
                
                except Exception as e:
                    print(f"Failed to upload {img_file}: {e}")



if __name__ == "__main__":
    client = MongoClient("mongodb+srv://yacinmontacer:7wyCP6HN7hqwsa6B@cluster0.jgqza.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client['image_database']  # Create or access the database
    collection = db['images']      # Create or access the collection


    """
    doc = collection.find_one({"class": "forest", "dataset_type": "train"})
    decoded_image = decode_image(doc["image"])
    cv2.imshow("Decoded Image", decoded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    train_images, train_labels = load_images_from_db('train')
    test_images, test_labels = load_images_from_db('test')

    #Encode labels to integers and convert to categorical (one-hot encoding)
    unique_labels = list(collection.distinct("class"))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    train_labels = tf.keras.utils.to_categorical([label_map[label] for label in train_labels], num_classes=len(unique_labels))
    test_labels = tf.keras.utils.to_categorical([label_map[label] for label in test_labels], num_classes=len(unique_labels))

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(unique_labels), activation='softmax'),
        ])
        
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    model.save('mongo_image_classifier.h5')
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


