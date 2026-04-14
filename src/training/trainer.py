import torch  # Import du noyau PyTorch pour la gestion des tenseurs et du calcul automatique des gradients
import os  # Import du module os pour gérer les fichiers et répertoires

class Trainer:
    def __init__(self, model, optimizer, criterion, device,checkpoint_dir="checkpoints"):
        self.model = model  # Le modèle PyTorch à entraîner
        self.optimizer = optimizer  # L'optimiseur utilisé pour mettre à jour les poids du modèle
        self.criterion = criterion  # La fonction de perte (critère) pour mesurer l'erreur
        self.device = device  # L'appareil de calcul (CPU ou GPU)
        self.checkpoint_dir = checkpoint_dir  # Dossier où sauvegarder les checkpoints
        self.best_acc = 0.0  # Meilleure précision observée jusqu'à présent
        
    def train_epoch(self, loader):
        self.model.train()  # Met le modèle en mode entraînement (active dropout/batchnorm si présents)
        total_loss = 0.0  # Variable pour accumuler la perte totale de l'époque
        correct = 0  # Variable pour compter le nombre de prédictions correctes
        for images, labels in loader:  # Boucle sur toutes les batches du DataLoader
            images, labels = images.to(self.device), labels.to(self.device)  # Envoie les données sur l'appareil de calcul
            self.optimizer.zero_grad()  # Réinitialise les gradients du pas précédent
            outputs = self.model(images)  # Passe les images dans le modèle pour obtenir les prédictions
            loss = self.criterion(outputs, labels)  # Calcule la perte entre les prédictions et les labels
            loss.backward()  # Rétropropagation : calcule les gradients
            self.optimizer.step()  # Met à jour les paramètres du modèle
            total_loss += loss.item()  # Ajoute la perte de la batch au total
            correct += (outputs.argmax(dim=1) == labels).sum().item()  # Compte les prédictions exactes
        return total_loss / len(loader), correct / len(loader.dataset)  # Retourne la perte moyenne et l'exactitude

    def evaluate(self, loader):
        self.model.eval()  # Met le modèle en mode évaluation (désactive dropout/batchnorm.adapt)
        total_loss = 0.0  # Variable pour accumuler la perte totale
        correct = 0  # Variable pour compter les bonnes prédictions
        with torch.no_grad():  # Désactive le calcul des gradients pour gagner du temps et de la mémoire
            for images, labels in loader:  # Boucle sur les batches de validation/test
                images, labels = images.to(self.device), labels.to(self.device)  # Envoie les données sur l'appareil
                outputs = self.model(images)  # Génère les sorties du modèle
                loss = self.criterion(outputs, labels)  # Calcule la perte
                total_loss += loss.item()  # Accumule la perte de la batch
                correct += (outputs.argmax(dim=1) == labels).sum().item()  # Compte les prédictions correctes
        return total_loss / len(loader), correct / len(loader.dataset)  # Retourne la perte moyenne et la précision

    def save_checkpoint(self, epoch, acc):
        if acc > self.best_acc:  # Si la précision actuelle est meilleure que la meilleure précision connue
            self.best_acc = acc  # Met à jour la meilleure précision
            os.makedirs(self.checkpoint_dir, exist_ok=True)  # Crée le dossier de checkpoint si nécessaire
            checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model.pth")  # Chemin du fichier de checkpoint
            torch.save({  # Sauvegarde les états nécessaires pour reprendre l'entraînement
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "accuracy": acc
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} with accuracy {acc:.4f}")  # Affiche un message de confirmation