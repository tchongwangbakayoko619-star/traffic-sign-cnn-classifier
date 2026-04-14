import torch  # Import de PyTorch pour charger le modèle et faire des calculs sur tenseurs
from PIL import Image  # Import de Pillow pour ouvrir et manipuler l'image
from src.models.cnn    import GTSRB_CNN  # Import de l'architecture CNN pour la prédiction
from src.data.dataset  import CLASS_NAMES  # Import des noms de classes du dataset GTSRB
from src.data.transforms import INFERENCE  # Import de la transformation d'inférence pour l'image

def predict(image_path: str, checkpoint_path="checkpoints/best_model.pth"):
    # Création du modèle et chargement du checkpoint sauvegardé
    model = GTSRB_CNN(num_classes=43)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # Mode évaluation : désactive dropout et n'accumule pas les gradients

    # Chargement de l'image, conversion en RGB et application des transformations
    image  = Image.open(image_path).convert("RGB")
    tensor = INFERENCE(image).unsqueeze(0)  # Ajoute une dimension batch

    # Prédiction sans calcul de gradient
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)  # Conversion des sorties en probabilités

    # Récupération de la classe la plus probable et de sa confiance
    confidence, idx = torch.max(probs, dim=1)
    print(f"Panneau détecté : {CLASS_NAMES[idx.item()]}")
    print(f"Confiance       : {confidence.item()*100:.2f}%")

if __name__ == "__main__":
    import sys
    predict(sys.argv[1])   # Utilisation : python predict.py chemin_vers_image.jpg