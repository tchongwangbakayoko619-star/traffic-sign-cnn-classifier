import torch  # Import de PyTorch pour charger le modèle et gérer les tenseurs
import yaml  # Import de YAML pour lire le fichier de configuration
from src.models.cnn       import GTSRB_CNN  # Import de l'architecture CNN définie pour GTSRB
from src.data.dataset     import get_dataloaders, CLASS_NAMES  # Import des dataloaders et des noms de classes
from src.utils.metrics    import evaluate_full  # Import de la fonction d'évaluation complète
from src.utils.visualize  import plot_confusion_matrix  # Import de la fonction de visualisation

# Lecture de la configuration depuis le fichier YAML
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

# Choix de l'appareil de calcul : GPU si disponible sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement des données de test uniquement (train et val sont ignorés ici)
_, _, test_loader = get_dataloaders(cfg["data"]["data_dir"],
                                    cfg["data"]["batch_size"])

# Instanciation du modèle CNN avec le nombre de classes du dataset
model = GTSRB_CNN(num_classes=43)
ckpt  = torch.load("checkpoints/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
# Déplacement du modèle vers l'appareil choisi
model.to(device)

if __name__ == '__main__':
    # Évaluation du modèle sur l'ensemble de test
    evaluate_full(model, test_loader, device, CLASS_NAMES)
    # Affichage de la matrice de confusion pour analyser les prédictions
    plot_confusion_matrix(model, test_loader, device, CLASS_NAMES)