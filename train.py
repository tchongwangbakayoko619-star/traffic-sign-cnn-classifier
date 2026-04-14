import torch  # Import de PyTorch pour la formation du modèle
import yaml  # Import de YAML pour charger la configuration
from src.models.cnn         import GTSRB_CNN  # Import de l'architecture CNN pour GTSRB
from src.data.dataset       import get_dataloaders  # Import de la fonction qui crée les dataloaders
from src.training.trainer   import Trainer  # Import de la classe Trainer qui gère l'entraînement et l'évaluation
from src.training.losses    import get_criterion  # Import de la fonction qui retourne la fonction de perte
from src.utils.visualize    import plot_training_curves  # Import pour tracer les courbes d'entraînement

# Lecture du fichier de configuration YAML
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

# Choix du device : GPU si disponible, sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# Chargement des dataloaders pour train, validation et test
train_loader, val_loader, test_loader = get_dataloaders(
    data_dir   = cfg["data"]["data_dir"],
    batch_size = cfg["data"]["batch_size"]
)

# Instanciation du modèle et de l'optimiseur
model     = GTSRB_CNN(num_classes=cfg["model"]["num_classes"])
optimizer = torch.optim.Adam(model.parameters(), lr    = cfg["training"]["learning_rate"], weight_decay = cfg["training"]["weight_decay"])
criterion = get_criterion(cfg["training"]["loss"])  # Sélection de la fonction de perte
trainer   = Trainer(model, optimizer, criterion, device,
                    checkpoint_dir=cfg["checkpoints"]["dir"])

# Listes pour conserver les courbes d'entraînement et de validation
train_accs, val_accs       = [], []
train_losses, val_losses   = [], []

if __name__ == '__main__':
    # Boucle d'entraînement sur le nombre d'époques défini
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t_loss, t_acc = trainer.train_epoch(train_loader)  # Entraîne une époque sur les données d'entraînement
        v_loss, v_acc = trainer.evaluate(val_loader)  # Évalue le modèle sur le jeu de validation
        trainer.save_checkpoint(epoch, v_acc)  # Sauvegarde le modèle si la précision de validation s'améliore

        # Sauvegarde des métriques pour tracer les courbes
        train_accs.append(t_acc);   val_accs.append(v_acc)
        train_losses.append(t_loss); val_losses.append(v_loss)

        # Affichage des performances pour chaque époque
        print(f"Epoch {epoch:02d} | " f"Train {t_acc*100:.2f}% loss={t_loss:.4f} | " f"Val {v_acc*100:.2f}% loss={v_loss:.4f}")

    # Tracer et sauvegarder les courbes d'entraînement et de validation
    plot_training_curves(train_accs, val_accs, train_losses, val_losses, save_path=f"{cfg['logs']['dir']}/curves.png")