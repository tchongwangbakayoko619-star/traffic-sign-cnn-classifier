from torchvision import transforms  # Import du module transforms de torchvision pour définir les transformations d'images

AUGMENTATION_HEAVY= transforms.Compose([  # Définition d'une transformation composée pour l'augmentation de données
    transforms.Resize((32, 32)),  # Redimensionner l'image à 32x32 pixels
    transforms.RandomRotation(15),  # Appliquer une rotation aléatoire de ±15 degrés pour l'augmentation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Appliquer une transformation affine aléatoire pour l'augmentation
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),  # Varier la luminosité et le contraste pour l'augmentation
    transforms.RadomGrayscale(p=0.1),  # Convertir en niveaux de gris avec une probabilité de 10% pour l'augmentation
    transforms.ToTensor(),  # Convertir l'image PIL en tenseur PyTorch
    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])  # Normaliser avec les moyennes et écarts-types calculés
])

AUGMENTATION_LIGHT = transforms.Compose([  # Définition d'une transformation composée pour une augmentation légère
    transforms.Resize((32, 32)),  # Redimensionner l'image à 32x32 pixels
    transforms.RandomHorizontalFlip(),  # Appliquer un retournement horizontal aléatoire pour l'augmentation
    transforms.ToTensor(),  # Convertir l'image PIL en tenseur PyTorch
    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])  # Normaliser avec les moyennes et écarts-types calculés
])

INFERENCE = transforms.Compose([  # Définition d'une transformation composée pour l'inférence
    transforms.Resize((32, 32)),  # Redimensionner l'image à 32x32 pixels
    transforms.ToTensor(),  # Convertir l'image PIL en tenseur PyTorch
    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])  # Normaliser avec les moyennes et écarts-types calculés
])