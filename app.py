from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
import base64
from src.models.cnn import GTSRB_CNN
from src.data.dataset import CLASS_NAMES, get_transforms

app = Flask(__name__)

# Configuration
MODEL_PATH = "checkpoints/best_model.pth"
NUM_CLASSES = 43
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle
def load_model():
    model = GTSRB_CNN(num_classes=NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

model = load_model()

# Transformation pour l'inférence
inference_transform = get_transforms(train=False)

@app.route('/')
def home():
    """Page d'accueil avec formulaire d'upload"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour faire des prédictions"""
    try:
        # Vérifier si un fichier a été uploadé
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400

        # Ouvrir l'image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Prétraiter l'image
        tensor = inference_transform(image).unsqueeze(0).to(device)

        # Faire la prédiction
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        # Convertir l'image en base64 pour l'affichage
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Résultat
        result = {
            'prediction': CLASS_NAMES[predicted_class.item()],
            'confidence': f"{confidence.item()*100:.2f}%",
            'class_id': int(predicted_class.item()),
            'image': img_str
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Endpoint de vérification de santé"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)