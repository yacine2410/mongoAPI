#module for basic CRUD operations
from pymongo import MongoClient
import cv2
import base64
import numpy as np
from datetime import datetime
from PIL import Image
import os

#CRUD Operations
#Upload individual image
def insert_image(image_path, collection):
    #Read the image using OpenCV
    image = cv2.imread(image_path)
    
    #Convert to binary format
    _, buffer = cv2.imencode('.jpg', image)
    image_binary = base64.b64encode(buffer).decode('utf-8')
    
    #Insert into MongoDB
    image_data = {
        "filename": image_path.split('/')[-1],
        "image": image_binary
    }
    collection.insert_one(image_data)
    print(f"Inserted {image_path} into MongoDB.")

#Upload multiple images
def upload_images(image_paths, collection):
    for image_path in image_paths:
        try:
            image = cv2.imread(image_path)
            _, buffer = cv2.imencode('.jpg', image)
            image_binary = base64.b64encode(buffer).decode('utf-8')
            image_data = {"filename": image_path.split('/')[-1], "image": image_binary}
            collection.insert_one(image_data)
            print(f"Uploaded: {image_path}")
        except Exception as e:
            print(f"Failed to upload {image_path}: {e}")

#get image by name
def retrieve_image_by_name(filename, collection):
    image_doc = collection.find_one({"filename": filename})
    if image_doc:
        image_binary = base64.b64decode(image_doc['image'])
        image_np = np.frombuffer(image_binary, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"No image found with filename: {filename}")
        return None

#Update image
def update_image(filename, new_image_path, collection):
    image = cv2.imread(new_image_path)
    _, buffer = cv2.imencode('.jpg', image)
    image_binary = base64.b64encode(buffer).decode('utf-8')

    result = collection.update_one(
        {"filename": filename},
        {"$set": {"image": image_binary}}
    )
    if result.matched_count > 0:
        print(f"Updated image: {filename}")
    else:
        print(f"No image found with filename: {filename}")

#Delete image
def delete_image(filename, collection):
    result = collection.delete_one({"filename": filename})
    if result.deleted_count > 0:
        print(f"Deleted image: {filename}")
    else:
        print(f"No image found with filename: {filename}")

#Clear collection
def clear_collection(collection):
    result = collection.delete_many({})
    print(f"Deleted {result.deleted_count} images from the collection.")

#Save image to PC
def download_image(filename, save_path, collection):
    image_doc = collection.find_one({"filename": filename})
    if image_doc:
        image_binary = base64.b64decode(image_doc['image'])
        image_np = np.frombuffer(image_binary, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        cv2.imwrite(save_path, image)
        print(f"Saved {filename} to {save_path}")
    else:
        print(f"No image found with filename: {filename}")

def decode_image(image_binary):
    img_data = base64.b64decode(image_binary)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image