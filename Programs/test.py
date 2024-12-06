import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
from Program.Utils.getData import Data 

def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for src, trg in data_loader:
            src = src.permute(0, 3, 1, 2).float()  # Convert to correct shape for model input
            trg = torch.argmax(trg, dim=1)  # Convert one-hot to integer labels

            outputs = model(src)
            probs = torch.softmax(outputs, dim=1)  # Get probabilities for AUC
            _, preds = torch.max(outputs, 1)  # Get predicted class labels

            all_preds.append(preds.cpu().numpy())
            all_labels.append(trg.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate all predictions, labels, and probabilities
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Check for unique labels and validate
    unique_labels = np.unique(all_labels)
    print(f"Unique labels in all_labels: {unique_labels}")

    NUM_CLASSES = 6  # Adjust according to your dataset
    if np.any(all_labels >= NUM_CLASSES) or np.any(all_labels < 0):
        raise ValueError(
            f"Invalid labels detected. Labels must be in the range [0, {NUM_CLASSES-1}]. Found: {unique_labels}"
        )

    # Compute AUC
    all_labels_onehot = np.eye(NUM_CLASSES)[all_labels]  # One-hot encode the labels for AUC calculation
    auc = roc_auc_score(all_labels_onehot, all_probs, multi_class='ovr', average='weighted')

    # Calculate other metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, auc, cm

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def main():
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6 

    # Paths to dataset
    aug_path = "D:/SEMESTER 3/IPSD-Assignment/W 10/Dataset/Augmented Images/FOLDS_AUG/"
    orig_path = "D:/SEMESTER 3/IPSD-Assignment/W 10/Dataset/Original Images/FOLDS/"

    # Initialize dataset and test loader
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    test_data = dataset.dataset_test
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Load pre-trained model with updated weights parameter
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)  # Adjust the classifier layer
    model.load_state_dict(torch.load("trained_modelmobilenet.pth"))
    model.eval()

    # Evaluate model on test data
    accuracy, precision, recall, f1, auc, cm = evaluate_model(model, test_loader)

    # Display evaluation results
    print("Evaluasi pada data test:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Visualize and save the confusion matrix as a heatmap
    class_names = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"]
    plot_confusion_matrix(cm, class_names, save_path="./confusion_matrix.png")

if __name__ == "__main__":
    main()