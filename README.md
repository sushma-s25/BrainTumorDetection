🧠 Brain Tumor Classification System
This project implements a Brain Tumor Classification System using Convolutional Neural Networks (CNNs) to detect tumor types from MRI images. It uses a custom-trained model and features an interactive Gradio UI for image-based predictions.

🔍 Features
✅ Classifies brain MRI scans into:

Glioma

Meningioma

No Tumor

Pituitary

✅ Built with TensorFlow/Keras for model architecture and training

✅ Uses OpenCV for image preprocessing

✅ Applies LabelBinarizer for multi-class output encoding

✅ Provides an intuitive Gradio interface for drag-and-drop predictions

🧠 How It Works
Load and preprocess brain MRI images (resized to 128x128).

Train a CNN model using layers like:

Conv2D

MaxPooling2D

Dropout

Predict tumor type from new MRI images using the trained model.

Display results with confidence scores via the Gradio interface.

📁 Requirements
Make sure you have Python 3.x installed and the following libraries:

pip install tensorflow opencv-python numpy scikit-learn gradio
If running in Google Colab, also install:

pip install google-colab
🚀 Usage
Option 1: Run in Google Colab
Upload your dataset to your Google Drive.

Mount Google Drive in your Colab notebook:

from google.colab import drive
drive.mount('/content/drive')
Ensure your dataset path is correctly referenced (e.g., /content/drive/MyDrive/brain_tumor_dataset).

Option 2: Run Locally
Clone/download the project and place the dataset in a directory:

brain_tumor_classification/
├── model/
├── dataset/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── main.py
└── ...
Run the script:

python main.py
🖼️ Gradio UI
Once the model is trained or loaded, Gradio will launch an interactive web interface for you to upload an image and view predictions:

import gradio as gr
gr.Interface(fn=predict_fn, inputs="image", outputs="label").launch()
