import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch

def plot_training_curves(train_accs, val_accs, train_losses, val_losses, save_path="logs/curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_accs, label="Train")
    ax1.plot(val_accs,   label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Époques")
    ax1.legend()

    ax2.plot(train_losses, label="Train")
    ax2.plot(val_losses,   label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Époques")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Courbes sauvegardées → {save_path}")
    plt.show()

def plot_confusion_matrix(model, loader, device, class_names, save_path="logs/confusion_matrix.png"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matrice de confusion")
    plt.ylabel("Vrai label")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Matrice sauvegardée → {save_path}")
    plt.show()