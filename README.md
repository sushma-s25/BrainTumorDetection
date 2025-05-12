
# 🧠 Brain Tumor Classification System

This project implements a **Brain Tumor Classification System** using **Convolutional Neural Networks (CNNs)** to detect tumor types from **MRI images**. It uses a custom-trained model and features an interactive **Gradio UI** for image-based predictions.

---

## 🔍 Features

* ✅ **Classifies brain MRI scans** into the following categories:

  * **Glioma**
  * **Meningioma**
  * **No Tumor**
  * **Pituitary**

* ✅ Built with **TensorFlow/Keras** for model architecture and training.

* ✅ Uses **OpenCV** for **image preprocessing**.

* ✅ Applies **LabelBinarizer** for multi-class output encoding.

* ✅ Provides an intuitive **Gradio interface** for **drag-and-drop predictions**.

---

## 🧠 How It Works

1. **Load and preprocess** brain MRI images (resized to 128x128 pixels).
2. **Train a CNN model** using layers like:

   * **Conv2D**: For feature extraction from MRI images.
   * **MaxPooling2D**: To reduce the spatial dimensions of the feature maps.
   * **Dropout**: To prevent overfitting during training.
3. **Predict tumor type** from new MRI images using the trained model.
4. **Display results** with confidence scores via the Gradio interface.

---

## 📁 Requirements

Make sure you have **Python 3.x** installed and the following libraries:

```bash
pip install tensorflow opencv-python numpy scikit-learn gradio
```

If running in **Google Colab**, also install:

```bash
pip install google-colab
```

---

## 🚀 Usage

### Option 1: Run in Google Colab

1. Upload your dataset to your **Google Drive**.

2. Mount Google Drive in your Colab notebook:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Ensure your dataset path is correctly referenced (e.g., `/content/drive/MyDrive/brain_tumor_dataset`).

---

### Option 2: Run Locally

1. Clone or download the project and place the dataset in a directory structure like this:

   ```plaintext
   brain_tumor_classification/
   ├── model/
   ├── dataset/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── notumor/
   │   └── pituitary/
   ├── main.py
   └── ...
   ```

2. Run the script:

   ```bash
   python main.py
   ```

---

## 🖼️ Gradio UI

Once the model is trained or loaded, **Gradio** will launch an interactive web interface where you can **upload an image** and view the predicted tumor type.

Here's the Gradio code to launch the interface:

```python
import gradio as gr

gr.Interface(fn=predict_fn, inputs="image", outputs="label").launch()
```

---

