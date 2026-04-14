import torch
from sklearn.metrics import f1_score, classification_report
import numpy as np

def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()

def evaluate_full(model, loader, device, class_names=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    f1  = f1_score(all_labels, all_preds, average="weighted")

    print(f"Accuracy : {acc*100:.2f}%")
    print(f"F1 Score : {f1:.4f}")

    if class_names:
        print("\n--- Rapport détaillé ---")
        print(classification_report(all_labels, all_preds,
                                    target_names=class_names))
    return acc, f1