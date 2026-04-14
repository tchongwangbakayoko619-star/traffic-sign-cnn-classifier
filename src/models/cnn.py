import torch.nn as nn  # Import de la bibliothèque de modules de PyTorch pour construire le réseau de neurones

class GTSRB_CNN(nn.Module):  # Définition de la classe GTSRB_CNN qui hérite de nn.Module
    def __init__(self, num_classes=43):
        super(GTSRB_CNN, self).__init__()
        
        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Convolution 2D avec 3 canaux d'entrée, 32 filtres, noyau de taille 3x3 et padding de 1
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.MaxPool2d(kernel_size=2),  # Max pooling avec un noyau de taille 2x2
            # Bloc 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Convolution 2D avec 32 canaux d'entrée, 64 filtres, noyau de taille 3x3 et padding de 1
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.MaxPool2d(kernel_size=2),  # Max pooling avec un noyau de taille 2x2
            # Bloc 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.MaxPool2d(kernel_size=2)  # Max pooling avec un noyau
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Aplatir les caractéristiques extraites en un vecteur
            nn.Linear(128*4*4, 256),  # Couche linéaire avec 128*4*4 entrées (taille des caractéristiques extraites) et 256 sorties
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.Dropout(0.5),  # Dropout avec un taux de 50% pour éviter le surapprentissage
            nn.Linear(256, num_classes)  # Couche linéaire finale avec 256 entrées et num_classes sorties (nombre de classes à prédire)
        )
        
    def forward(self, x):  # Définition de la méthode forward pour la propagation avant du réseau
        x = self.features(x)  # Passer l'entrée à travers les couches de caractéristiques
        x = self.classifier(x)  # Passer les caractéristiques extraites à travers les couches de classification
        return x  # Retourner les prédictions finales du réseau