# main.py

"""
Face Parsing App
This application allows users to upload a photo of a person and perform semantic segmentation of the face.
"""

import streamlit as st
from PIL import Image
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt

def load_model():
    """
    Load the face parsing model and image processor.

    Returns:
        Tuple: (image_processor, model, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    model.to(device)
    return image_processor, model, device

# Load models
image_processor, model, device = load_model()

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Resize the input image to match the model's input size.

    Args:
        img (Image.Image): Input image.

    Returns:
        Image.Image: Resized image.
    """
    img = img.resize((512, 512))
    return img

def process_and_display(image: Image.Image):
    """
    Run inference on the input image and display the results.

    Args:
        image (Image.Image): Input image.
    """
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(labels)
    ax.axis("off")
    st.pyplot(fig)

st.title("Face Parsing App")
uploaded_file = st.file_uploader(label="Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocessed_image = preprocess_image(image)
    result = st.button("Process Image")
    if result:
        process_and_display(preprocessed_image)
