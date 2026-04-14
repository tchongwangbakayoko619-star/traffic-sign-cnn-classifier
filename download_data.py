from torchvision import datasets

print("Téléchargement GTSRB en cours...")
print("Cela peut prendre 5 à 10 minutes selon votre connexion.")
print("")

# Télécharge automatiquement dans data/raw/
train = datasets.GTSRB(root="data/raw", split="train", download=True)
test  = datasets.GTSRB(root="data/raw", split="test",  download=True)

print("")
print(f"Train : {len(train)} images")
print(f"Test  : {len(test)}  images")
print("")
print("Téléchargement terminé ! Dataset disponible dans data/raw/")