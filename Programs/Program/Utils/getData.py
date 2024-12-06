import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, base_folder_aug, base_folder_orig):
        """
        :param base_folder_aug: Path folder untuk Augmented Images
        :param base_folder_orig: Path folder untuk Original Images
        """
        self.dataset_aug = []
        self.dataset_train = []
        self.dataset_test = []
        self.dataset_valid = []
        onehot = np.eye(6)  # One-hot encoding untuk 6 kelas
        
        # Load data dari Augmented Images (Train saja)
        for fold_num in range(1, 6):
            aug_folder = os.path.join(base_folder_aug, f"fold{fold_num}_AUG/Train/")
            for class_idx, class_name in enumerate(os.listdir(aug_folder)):
                class_folder = os.path.join(aug_folder, class_name)
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image = cv.resize(cv.imread(img_path), (32, 32)) / 255
                    self.dataset_aug.append([image, onehot[class_idx]])
        
        # Load data dari Original Images (Train, Test, Valid)
        for fold_num in range(1, 6):
            fold_folder = os.path.join(base_folder_orig, f"fold{fold_num}/")
            
            # Load Train
            train_folder = os.path.join(fold_folder, "Train/")
            for class_idx, class_name in enumerate(os.listdir(train_folder)):
                class_folder = os.path.join(train_folder, class_name)
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image = cv.resize(cv.imread(img_path), (32, 32)) / 255
                    self.dataset_train.append([image, onehot[class_idx]])

            # Load Test
            test_folder = os.path.join(fold_folder, "Test/")
            for class_idx, class_name in enumerate(os.listdir(test_folder)):
                class_folder = os.path.join(test_folder, class_name)
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image = cv.resize(cv.imread(img_path), (32, 32)) / 255
                    self.dataset_test.append([image, onehot[class_idx]])

            # Load Valid
            valid_folder = os.path.join(fold_folder, "Valid/")
            for class_idx, class_name in enumerate(os.listdir(valid_folder)):
                class_folder = os.path.join(valid_folder, class_name)
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image = cv.resize(cv.imread(img_path), (32, 32)) / 255
                    self.dataset_valid.append([image, onehot[class_idx]])
        
        # Debug output
        print(f"Augmented Images (Train): {len(self.dataset_aug)}")
        print(f"Original Images (Train): {len(self.dataset_train)}")
        print(f"Original Images (Test): {len(self.dataset_test)}")
        print(f"Original Images (Valid): {len(self.dataset_valid)}")

    def __len__(self):
        """Mengembalikan jumlah data di Augmented Images (default)."""
        return len(self.dataset_aug)

    def __getitem__(self, idx):
        """
        :param idx: Index data
        :return: Tuple (image, label) dalam format tensor
        """
        features, label = self.dataset_aug[idx]
        return (torch.tensor(features, dtype=torch.float32).permute(2, 0, 1),  # Convert to CHW format
                torch.tensor(label, dtype=torch.float32))


if __name__ == "__main__":
    
    aug_path = "D:/SEMESTER 3/IPSD-Assignment/W 10/Dataset/Augmented Images/FOLDS_AUG/"
    orig_path = "D:/SEMESTER 3/IPSD-Assignment/W 10/Dataset/Original Images/FOLDS/"
    # Inisialisasi Data
    data = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)