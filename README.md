# Streamlit Face Parsing App

This application allows users to upload an image of a person and perform semantic segmentation of the face.

## Features

- Upload an image: Users can upload a photo of a person from their device.
- Process Image: After uploading an image, users can click on the "Process Image" button to run the semantic segmentation process.
- Display Results: The segmented face image is displayed below the uploaded photo.

## Technologies Used

- Streamlit: Used for creating the web application interface.
- PyTorch: Utilized for running inference with the face parsing model.
- Transformers: Used for loading the pre-trained face parsing model.
- PIL (Python Imaging Library): Used for image preprocessing.
- Matplotlib: Used for visualizing the segmented face image.

## Installation

To run the application locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/CatPawsCoder/streamlit_transform_excercise.git
