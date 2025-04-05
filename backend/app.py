from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import json
import random
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Step 2: Generate tags using Gemini API (real call)
def get_tags_for_image(image_filename):
    image_path = os.path.join(IMAGE_FOLDER, image_filename)

    prompt = (
        "Add tags to this image so we can search them. Add tags based on: "
        "if picture is of a person: facial expression, appearance, body position, what they are doing, gender, and name of the person if known, also take help of web search to find who they are why they are relevant and what do they do, also movie name if the shown picture is from a movie scene and also dialogs of this scene if available. "
        "If picture is of a cartoon: name of the character, show/movie/series name, colour, facial expression, appearance, body position, gender, if cartoon is of an animal or birds and the general specie name and class of organism they belong, their voices (eg: cow = moo). "
        "Also add tags that describe what’s happening in the image (for example: screaming, dancing, fighting, laughing etc). "
        "Return the tags in form of an array: [\"tag1\", \"tag2\", ...]"
    )

    image = Image.open(image_path)
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image]
        )
        tags = json.loads(response.text)
        return tags
    except Exception as e:
        print("Gemini API error:", e)
        return []

# Step 3: Assign tags to clusters and save mapping
def tag_images():
    clusters = cluster_images()
    tag_mapping = load_tag_mapping()

    for cluster, images in clusters.items():
        if all(img in tag_mapping for img in images):
            continue
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

@app.get("/")
def root():
    return {"message": "FastAPI Image Search API is running."}

@app.get("/search")
def search(description: str = Query(..., title="Search Description")):
    images = search_images(description)
    return {"images": images}

@app.post("/process")
def process_images():
    tag_images()
    return {"message": "Images processed and tagged successfully."}
