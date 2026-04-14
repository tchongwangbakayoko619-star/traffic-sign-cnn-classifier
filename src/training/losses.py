import torch.nn as nn  # Import de la bibliothèque de modules de PyTorch pour construire les fonctions de perte

def get_criterion(name="cross_entropy"):
    if name=="cross_entropy":  # Si le nom de la fonction de perte est "cross_entropy"
        return nn.CrossEntropyLoss()  # Retourner une instance de la fonction de perte CrossEntropyLoss
    else:
        raise ValueError(f"Unsupported loss function: {name}")  # Lever une exception si le nom n'est pas supporté