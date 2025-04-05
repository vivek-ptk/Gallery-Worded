from fastapi import FastAPI, Query
from typing import List
import os
import json
import random
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

app = FastAPI()

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Paths
IMAGE_FOLDER = "backend/images"
TAG_MAPPING_FILE = "backend/tag_mapping.json"

# Load or initialize tag mapping
def load_tag_mapping():
    if os.path.exists(TAG_MAPPING_FILE):
        with open(TAG_MAPPING_FILE, "r") as f:
            return json.load(f)
    return {}

def save_tag_mapping(mapping):
    with open(TAG_MAPPING_FILE, "w") as f:
        json.dump(mapping, f)

# Extract image embeddings
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.cpu().numpy().flatten()

# Step 1: Cluster images
def cluster_images():
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('png', 'jpg', 'jpeg'))]
    image_paths = [os.path.join(IMAGE_FOLDER, f) for f in image_files]
    embeddings = np.array([get_image_embedding(img) for img in image_paths])
    
    num_clusters = min(5, len(image_files))  # Adjust number of clusters dynamically
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    clusters = {}
    for img, label in zip(image_files, labels):
        clusters.setdefault(label, []).append(img)
    
    return clusters

# Step 2: Generate tags using an external LLM (Mocked for now)
def get_tags_for_image(image):
    return ["tag1", "tag2", "tag3"]  # Mocked response from LLM

# Step 3: Assign tags to clusters and save mapping
def tag_images():
    clusters = cluster_images()
    tag_mapping = {}
    
    for cluster, images in clusters.items():
        representative_image = random.choice(images)
        tags = get_tags_for_image(representative_image)
        for img in images:
            tag_mapping[img] = tags
    
    save_tag_mapping(tag_mapping)

def search_images(description):
    tag_mapping = load_tag_mapping()
    words = description.lower().split()
    matching_images = set()
    
    for img, tags in tag_mapping.items():
        if any(word in tags for word in words):
            matching_images.add(img)
    
    return list(matching_images)

@app.get("/search")
def search(description: str = Query(..., title="Search Description")):
    images = search_images(description)
    return {"images": images}

@app.post("/process")
def process_images():
    tag_images()
    return {"message": "Images processed and tagged successfully."}
