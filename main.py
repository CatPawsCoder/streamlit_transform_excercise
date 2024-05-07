import io
import streamlit as st
from PIL import Image
import requests
import numpy as np
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt

# Load models
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((512, 512))  # Resize image to match model input size
    return img

# Function to run inference and visualize results
def process_and_display(image):
    # Run inference
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits,
                                                  size=image.size[::-1], 
                                                  mode='bilinear',
                                                  align_corners=False)
    labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

      # Visualize results
    fig, ax = plt.subplots()
    ax.imshow(labels)
    ax.axis('off')
    st.pyplot(fig)

# Main Streamlit app
st.title('Face Parsing App')

uploaded_file = st.file_uploader(label='Upload an image', type=['jpg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and process the image
    preprocessed_image = preprocess_image(image)
    result = st.button('Process Image')
    if result:
        process_and_display(preprocessed_image)
