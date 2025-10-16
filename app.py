"""
FastAPI Backend for Diabetic Retinopathy Classification using ONNX Runtime
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import os
import time
from typing import List
import uvicorn
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms

app = FastAPI(
    title="Diabetic Retinopathy Classification API",
    description="CNN-based retinal image analysis (DR vs No_DR) using ONNX Runtime",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    print("Warning: Static directory not found")

# Global variables
onnx_session = None
model_loaded = False
class_names = ["DR", "No_DR"]  # 0=DR, 1=No_DR


def load_onnx_model():
    """Load the ONNX model"""
    global onnx_session, model_loaded
    
    try:
        # Try to find the model file
        possible_paths = ["retinopathy_model.onnx"]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print(f"ERROR: No ONNX model file found!")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in directory: {os.listdir('.')}")
            print(f"Looking for: {possible_paths}")
            return
        
        print(f"Loading ONNX model from: {model_path}")
        
        # Create inference session with ONNX runtime
        onnx_session = ort.InferenceSession(model_path)
        
        model_loaded = True
        print("ONNX model loaded and ready!")
        
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        model_loaded = False


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image to match the input expected by the model"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use the same transforms as in the training (without augmentation)
        transform = transforms.Compose([
            transforms.Resize((255, 255)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)
        
        # Convert to numpy and ensure float32 type for ONNX
        return image_tensor.numpy().astype(np.float32)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("Starting Diabetic Retinopathy Classification API (ONNX)")
    print("=" * 60)
    load_onnx_model()
    print("=" * 60)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Diabetic Retinopathy API</title></head>
            <body style="font-family: Arial; padding: 40px; text-align: center;">
                <h1>Diabetic Retinopathy Classification API (ONNX)</h1>
                <p>Binary Classification: DR vs No_DR</p>
                <p>API is running! Visit <a href="/docs">/docs</a> for documentation.</p>
                <p><a href="/health">Check Health Status</a></p>
            </body>
        </html>
        """)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "backend": "ONNX Runtime",
        "model_architecture": "CNN_Retino (4 Conv + 2 FC)",
        "num_classes": 2,
        "classes": class_names,
        "timestamp": time.time()
    }


@app.post("/predict")
async def predict_retinopathy(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        
        # Run inference with ONNX Runtime
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        
        outputs = onnx_session.run([output_name], {input_name: image_array})[0]
        
        # Convert log_softmax to probabilities
        probabilities = np.exp(outputs)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][predicted_class]
        
        severity_levels = {
            0: "Positive diagnosis. Retinal abnormalities consistent with Diabetic Retinopathy were detected.",
            1: "Negative diagnosis. No clear signs of Diabetic Retinopathy were detected in the analysis."
        }
        
        results = {
            "predicted_class": int(predicted_class),
            "predicted_class_name": class_names[predicted_class],
            "confidence": round(float(confidence) * 100, 2),
            "all_probabilities": {
                class_names[i]: round(float(probabilities[0][i]) * 100, 2)
                for i in range(len(class_names))
            },
            "diagnosis": get_diagnosis(predicted_class),
            "recommendations": get_recommendations(predicted_class),
            "severity_level": severity_levels.get(int(predicted_class), "Assessment not available.")
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def get_diagnosis(class_idx: int) -> str:
    diagnoses = {
        0: "Diabetic Retinopathy Detected",
        1: "No Diabetic Retinopathy Detected"
    }
    return diagnoses.get(class_idx, "Unknown")


def get_recommendations(class_idx: int) -> List[str]:
    recommendations = {
        0: [
            "⚠️ Diabetic retinopathy detected - Seek medical attention",
            "Schedule an appointment with an ophthalmologist immediately",
            "Maintain strict blood sugar control",
            "Monitor blood pressure and cholesterol levels",
            "Follow up regularly for progression monitoring"
        ],
        1: [
            "✓ No signs of diabetic retinopathy detected",
            "Continue regular eye exams as recommended",
            "Maintain good blood sugar control",
            "Keep managing diabetes effectively",
            "Schedule annual diabetic eye examination"
        ]
    }
    return recommendations.get(class_idx, ["Consult with healthcare provider"])


@app.get("/model-info")
async def get_model_info():
    if not model_loaded:
        return {"error": "Model not loaded"}
    
    return {
        "model_loaded": model_loaded,
        "backend": "ONNX Runtime",
        "class_names": class_names,
        "model_architecture": "CNN_Retino",
        "layers": {
            "conv_layers": 4,
            "fc_layers": 2,
            "filters": [8, 16, 32, 64],
            "fc_neurons": [100, 2]
        },
        "input_size": "255x255",
        "num_classes": 2,
        "training_accuracy": "94%",
        "test_accuracy": "93%"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
