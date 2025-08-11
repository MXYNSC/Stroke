import os
import cv2
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_auc_score
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE
import pandas as pd
import config as C

class CustomDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]
        
        return data, label


def Get_Datas_TXT(path):
    train_x = []
    train_y = []
    test_size=0.3
    with open(path, 'r', encoding="utf-8") as f:
        data = f.readlines()
    for line in data:
        datas = line.strip().split('\t')[0]
        label = int(line.strip().split('\t')[1])
        lines = [int(x) for x in datas.strip().split(',')]
        dd = [0 if ddd=='' else ddd for ddd in lines]
        train_x.append(lines)
        train_y.append(label)
        
    # smote = SMOTE(random_state=42)
    # train_x, train_y = smote.fit_resample(train_x, train_y)
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=test_size,random_state=42, shuffle=True, stratify=train_y)
    return X_train, X_test, y_train, y_test


def Get_Datas_CSV(csv_path):
    test_size=C.test_size
    df = pd.read_csv(csv_path)

    train_x = df.iloc[:, :-1].values
    train_y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        train_x, train_y, test_size=test_size, random_state=2025, shuffle=True, stratify=train_y
    )

    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))

    num_classes = conf_matrix.shape[0]  
    class_names = [str(i) for i in range(num_classes)]

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap=plt.cm.Blues,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={'size': 20}
    )

    plt.xlabel('Predicted Labels', fontsize=20)
    plt.ylabel('True Labels', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    full_path = os.path.join(f'{C.outpath}', 'confusion_matrix.png')
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {full_path}")


def plot_loss_curve(loss_history):
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, label='Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    full_path = os.path.join(f'{C.outpath}', 'loss.png')
    plt.savefig(full_path, dpi=300)
    print(f"Loss curve saved to: {full_path}")


def plot_accuracy_curve(train_accuracy_list, test_accuracy_list):
    epochs = range(1, len(train_accuracy_list)+1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy_list, label='Train Accuracy')
    plt.plot(epochs, test_accuracy_list, label='Test Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    full_path = os.path.join(f'{C.outpath}', 'accuracy_plot.png')
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"Accuracy curve saved to: {full_path}")


