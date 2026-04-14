from torchvision import datasets , transforms  # Import des modules datasets et transforms de torchvision pour gérer les données d'images
from torch.utils.data import DataLoader,random_split  # Import de DataLoader pour charger les données en batches et random_split pour diviser les ensembles de données

# Liste des noms des classes correspondant aux panneaux de signalisation du dataset GTSRB
CLASS_NAMES = [
    "Limite 20",        # Limite de vitesse à 20 km/h
    "Limite 30",        # Limite de vitesse à 30 km/h
    "Limite 50",        # Limite de vitesse à 50 km/h
    "Limite 60",        # Limite de vitesse à 60 km/h
    "Limite 70",        # Limite de vitesse à 70 km/h
    "Limite 80",        # Limite de vitesse à 80 km/h
    "Fin limite 80",    # Fin de la limite de vitesse 80 km/h
    "Limite 100",       # Limite de vitesse à 100 km/h
    "Limite 120",       # Limite de vitesse à 120 km/h
    "Dépassement interdit", # Interdiction de dépasser
    "Camions interdits", # Interdiction pour les camions
    "Priorité carrefour", # Priorité au carrefour
    "Route prioritaire", # Route prioritaire
    "Cédez le passage", # Cédez le passage
    "Stop",             # Panneau stop
    "Circulation interdite", # Circulation interdite
    "Camions interdits", # Interdiction pour les camions (dupliqué?)
    "Entrée interdite", # Entrée interdite
    "Danger",           # Danger général
    "Virage gauche",    # Virage à gauche
    "Virage droit",     # Virage à droite
    "Virages",          # Virages successifs
    "Dos d'âne",        # Dos d'âne
    "Chaussée glissante", # Chaussée glissante
    "Rétrécissement",   # Rétrécissement de chaussée
    "Travaux",          # Travaux
    "Feux tricolores",  # Feux tricolores
    "Piétons",          # Passage piétons
    "Enfants",          # Attention enfants
    "Cyclistes",        # Passage cyclistes
    "Verglas",          # Risque de verglas
    "Animaux",          # Animaux sauvages
    "Fin restrictions", # Fin des restrictions
    "Tourner droite",   # Obligation de tourner à droite
    "Tourner gauche",   # Obligation de tourner à gauche
    "Tout droit",       # Tout droit
    "Droit ou droite",  # Tout droit ou à droite
    "Droit ou gauche",  # Tout droit ou à gauche
    "Serrez droite",    # Serrez à droite
    "Serrez gauche",    # Serrez à gauche
    "Rond-point",       # Rond-point
    "Dépas. autorisé",  # Dépassement autorisé
    "Dépas. camions ok" # Dépassement autorisé pour les camions
    ]

def get_transforms(train=True):  # Définition de la fonction pour obtenir les transformations, train=True pour les données d'entraînement
    if train:  # Si c'est pour l'entraînement, appliquer des augmentations
        return transforms.Compose([  # Composer plusieurs transformations
            transforms.Resize((32, 32)),  # Redimensionner l'image à 32x32 pixels
            transforms.RandomRotation(10),  # Appliquer une rotation aléatoire de ±10 degrés pour l'augmentation
            transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Varier la luminosité et le contraste pour l'augmentation
            transforms.ToTensor(),  # Convertir l'image PIL en tenseur PyTorch
            transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])  # Normaliser avec les moyennes et écarts-types calculés
        ])
    else:  # Sinon, pour validation/test, pas d'augmentation
        return transforms.Compose([  # Composer les transformations simples
            transforms.Resize((32, 32)),  # Redimensionner à 32x32
            transforms.ToTensor(),  # Convertir en tenseur
            transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])  # Normaliser
        ])
        
def get_dataloaders(data_dir="data/raw",batch_size=64):  # Définition de la fonction pour obtenir les dataloaders avec répertoire par défaut et taille de batch
    train_set= datasets.GTSRB(root=data_dir,split="train",  # Charger l'ensemble d'entraînement du dataset GTSRB
                              transform=get_transforms(True),download=True)  # Appliquer les transformations d'entraînement et télécharger si absent
    test_set= datasets.GTSRB(root=data_dir,split="test",  # Charger l'ensemble de test
                             transform=get_transforms(False),download=True)  # Appliquer les transformations de test et télécharger
    # Séparer le train en train et validation  # Commentaire existant pour diviser l'ensemble d'entraînement
    val_size= int(0.2*len(train_set))  # Calculer 20% de la taille de train_set pour la validation
    train_size= len(train_set)-val_size  # Calculer la taille restante pour l'entraînement
    train_set, val_set= random_split(train_set,[train_size,val_size])  # Diviser aléatoirement train_set en train et val
    
    train_loader= DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)  # Créer DataLoader pour train avec mélange et 2 workers
    val_loader= DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=2)  # Créer DataLoader pour validation sans mélange
    test_loader= DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=2)  # Créer DataLoader pour test sans mélange

    return train_loader, val_loader, test_loader  # Retourner les trois DataLoaders