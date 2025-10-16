# Retinopathy AI Diabetic Retinopathy Detection using CNN (PyTorch + ONNX)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red?logo=pytorch)
![ONNX](https://img.shields.io/badge/ONNX-Model-lightblue?logo=onnx)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)
![GitHub Pages](https://img.shields.io/badge/Frontend-GitHub%20Pages-black?logo=github)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A deep learning project that detects **Diabetic Retinopathy** from retinal fundus images using a **Convolutional Neural Network (CNN)**.  
The model is trained in **PyTorch**, exported to **ONNX** for lightweight inference, and served through a **FastAPI backend** with a minimal **HTML frontend**.

---

## Project Overview

Diabetic Retinopathy (DR) is one of the leading causes of blindness worldwide.  
Early detection can significantly reduce vision loss.  
This project automates the detection process by classifying retinal images into different stages of DR severity.

![Retina Example](https://domf5oio6qrcr.cloudfront.net/medialibrary/14372/0b26ebff-2421-4333-aedc-bb17dfc37541.jpg)

---

## Tech Stack

- **Model:** PyTorch CNN  
- **Inference:** ONNX Runtime  
- **Backend:** FastAPI  
- **Frontend:** HTML/CSS (Static, hosted via GitHub Pages)  
- **Deployment:**  
  - Frontend → GitHub Pages  
  - Backend → Render / Hugging Face Spaces  

---

## Project Structure

```

retinopathy-ai/
│
├── backend/
│   ├── app.py                  # FastAPI backend serving ONNX model
│   ├── retinopathy_model.onnx  # Exported model for inference
│   └── requirements.txt
│
├── frontend/
│   ├── index.html              # Simple UI for image upload and prediction
│   └── images/                 # Sample test images
│
├── diagnosis-of-diabetic-retinopathy-by-pytorch.ipynb  # Model training notebook
└── README.md

````

---

## How It Works

1. **Dataset Loading:** Retinal images are loaded and organized into train, validation, and test sets.  
2. **Preprocessing:** Normalization and resizing ensure uniform CNN input.  
3. **Model Architecture:** A CNN trained to classify DR severity stages.  
4. **Training & Evaluation:**  
   - Optimizer: Adam  
   - Loss Function: CrossEntropy  
   - Metrics: Accuracy  
5. **Export:**  
   The trained PyTorch model is exported as .pt model.  
6. **Deployment:**  
   FastAPI backend serves predictions → HTML frontend displays results.

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/Talnz007/retinopathy-ai.git
cd retinopathy-ai/backend

# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn app:app --reload
````

Then open `frontend/index.html` in your browser.
Set the API endpoint in the script to:

```
http://127.0.0.1:8000/predict
```

---

## Deployment

### **Frontend**

Hosted on **GitHub Pages**:
[https://Talnz007.github.io/retinopathy-ai](https://Talnz007.github.io/retinopathy-ai)

---

## Results Snapshot

* **Model Type:** Custom CNN
* **Framework:** PyTorch
* **Export Format:** ONNX
* **Goal:** Classify retinal images into diabetic retinopathy stages
* **Outcome:** High validation accuracy and consistent generalization across test images.

---

## Author

**[Talha Niazi](https://github.com/Talnz007)**
B.S. Artificial Intelligence | Tech Enthusiast | Project-Based Learner

> *"Ship fast. Learn faster."*

---

## License

This project is licensed under the **MIT License** feel free to fork, modify, and use it for educational or research purposes.

---
