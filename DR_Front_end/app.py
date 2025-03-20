import os
import numpy as np
import joblib
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import load_model, Model

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load necessary models and scalers
try:
    selected_feature_indices = np.load('selected_feature_indices_2.npy')
    scaler = joblib.load('standard_scaler.pkl')
    model = load_model('optimized_extracted_model_2.h5')
    model.compile()
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    selected_feature_indices = None
    scaler = None
    model = None

# Load VGG16 model for feature extraction
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=vgg16_model.input, outputs=vgg16_model.output)

# Load Retina Classifier
class RetinaClassifier(nn.Module):
    def __init__(self):
        super(RetinaClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (224 // 4) * (224 // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load Retina Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retina_model = RetinaClassifier().to(device)
retina_model.load_state_dict(torch.load("retina_classifier.pth", map_location=device))
retina_model.eval()

# Define transformation for retina classifier
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def is_retina_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = retina_model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item() == 0

def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = feature_extractor.predict(img_array)
        features_flat = features.reshape((features.shape[0], -1))
        features_scaled = scaler.transform(features_flat)
        selected_features = features_scaled[:, selected_feature_indices]
        return selected_features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def get_eye_health_tips(pred_class):
    tips = {
        0: "Your eyes are healthy! Maintain a balanced diet and regular checkups.",
        1: "Mild DR detected. Consider an eye exam and monitor your blood sugar levels.",
        2: "Moderate DR detected. Schedule a consultation with an eye specialist.",
        3: "Severe DR detected! Immediate medical intervention is recommended.",
        4: "Proliferative DR detected! Seek urgent treatment to prevent vision loss."
    }
    return tips.get(pred_class, "No specific advice available.")

@app.route("/")
def home():
    return render_template("h.html")

#@app.route("/questionnaire", methods=["GET", "POST"])
#def questionnaire():
#    if request.method == "POST":
        # Collect answers correctly
#        answers = [
 #           request.form.get("blurry_vision", "0"),
#            request.form.get("floaters", "0"),
#            request.form.get("night_vision", "0"),
#            request.form.get("diabetes", "0"),
#            request.form.get("eye_pain", "0"),
#            request.form.get("fluctuating_vision", "0")
#        ]

        # Convert answers to numerical values ("yes" -> 1, "no" -> 0)
#        severity_score = sum(1 for ans in answers if ans == "yes")

#        if severity_score < 2:
#            return render_template("question.html", message="No severe symptoms detected. Prediction not required.")

#        session['allow_prediction'] = True
#        return redirect(url_for("predict"))

#    return render_template("question.html")

@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():
    if request.method == "POST":
        # Get all responses
        responses = {
            "blurry_vision": request.form.get("blurry_vision"),
            "floaters": request.form.get("floaters"),
            "night_vision": request.form.get("night_vision"),
            "diabetes": request.form.get("diabetes"),
            "eye_pain": request.form.get("eye_pain"),
            "fluctuating_vision": request.form.get("fluctuating_vision"),
        }

        # Define strong indicators of DR
        critical_symptoms = ["diabetes", "floaters", "fluctuating_vision"]

        # Check if the user has at least ONE critical symptom
        has_critical_symptom = any(responses[symptom] == "yes" for symptom in critical_symptoms)

        if has_critical_symptom:
            session['allow_prediction'] = True
            return redirect(url_for("predict"))
        else:
            return render_template("question.html", message="No high-risk symptoms detected. Prediction is not required.")

    return render_template("question.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not session.get('allow_prediction', False):
        return redirect(url_for("questionnaire"))
    
    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("index.html", prediction="No Image Selected", file_name=None)
        
        file = request.files["image"]
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        if not is_retina_image(file_path):
            os.remove(file_path)
            return render_template("index.html", prediction="Invalid Image! Please upload a retina image.", file_name=file.filename)
        
        try:
            features = extract_features(file_path)
            if features is None:
                os.remove(file_path)
                return render_template("index.html", prediction="Feature extraction failed!", file_name=file.filename)
            predictions = model.predict(features)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            health_tips = get_eye_health_tips(predicted_class)
            os.remove(file_path)
            return render_template("index.html", prediction=f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f}%)", health_tips=health_tips, file_name=file.filename)
        except Exception as e:
            return render_template("index.html", prediction=f"Error: {str(e)}", file_name=file.filename)
    
    return render_template("index.html", prediction=None, file_name=None)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
