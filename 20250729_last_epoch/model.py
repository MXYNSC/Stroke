from util import plot_confusion_matrix, plot_accuracy_curve, plot_loss_curve
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import datetime
import os
from itertools import combinations
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from torchmetrics.classification import Accuracy
import config as C
import logging


class EAPCR(nn.Module):

        
    def __init__(self, num_embed=0, embed_dim=0, dropout_prob=0, device='CPU'):
        super().__init__()

        self.device = device

        size, new_size,l, m ,= 12, 12, 4, 3

        T = self.Generator_matrix(size, new_size, l, m)
        self.T = torch.tensor(T, dtype=torch.float32).to(self.device)
        np.savetxt('T_numpy.txt', T, fmt='%d')

        self.embedding = nn.Embedding(num_embed, embed_dim)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.conv11 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.conv12 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        # self.conv13 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.conv14 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv21 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.conv22 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        # self.conv23 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.conv24 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc =  nn.Sequential(
                                nn.Linear(3872, 128),
                                nn.ReLU(),
                                nn.Dropout(dropout_prob),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Dropout(dropout_prob),
                                # nn.Linear(128, 128),
                                # nn.ReLU(),
                                # nn.Dropout(dropout_prob),
                                nn.Linear(128, 2)
                                )

        self.res = nn.Sequential(
                                nn.Linear(1536, 128),
                                nn.ReLU(),
                                nn.Dropout(dropout_prob),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Dropout(dropout_prob),
                                # nn.Linear(128, 128),
                                # nn.ReLU(),
                                # nn.Dropout(dropout_prob),
                                nn.Linear(128, 2)
                                )

        self.weight = nn.Parameter(torch.tensor(0.0))

        self.to(self.device)

        self.total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {self.total_params}")
        logging.info(f'Total number of parameters: {self.total_params}')


    def forward(self, x):

        x = x.long()
        E = self.embedding(x)

        R = E.reshape(E.size(0), -1)
        R = self.res(R)

        ET = E.transpose(1, 2)
        A = torch.matmul(E, ET)
        PA = torch.matmul(self.T, A)
        PA = torch.matmul(PA, self.T.transpose(0,1))
        # PA = torch.matmul(PA, self.T)
        PA = self.tanh(PA).unsqueeze(1)
        A = self.tanh(A).unsqueeze(1)
        
        C = self.relu(self.conv11(A))
        C = self.pool1(C)
        # C = self.relu(self.conv12(C))
        # C = self.pool1(C)
        # C = self.relu(self.conv13(C))
        # C = self.pool1(C)
        # C = self.relu(self.conv14(C))
        # C = self.pool1(C) 
        
        CC = self.relu(self.conv21(PA))
        CC = self.pool2(CC)
        # CC = self.relu(self.conv22(CC))
        # CC = self.pool2(CC)
        # CC = self.relu(self.conv23(CC))
        # CC = self.pool2(CC)
        # CC = self.relu(self.conv24(CC))
        # CC = self.pool2(CC)
        
        C_cat = torch.cat((C, CC), dim=1)
        C_cat = C_cat.reshape(C_cat.size(0), -1)
        C_cat = self.fc(C_cat)
        
        W = torch.sigmoid(self.weight)
        Output = W * C_cat + (1 - W) * R

        return Output


    def train_model(self, train_loader, test_loader, epochs=10, lr=1):
        
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(self.parameters(), lr=lr)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        loss_epoch_list = []
        train_accuracy_list = []
        test_accuracy_list = []
        test_recall_list = []
        test_precision_list = []
        test_F1_list = []
        
        for epoch in range(epochs):

            self.train()
            loss_epoch = 0
            
            for batch_idx, (train_x, train_y) in enumerate(train_loader):

                train_x = train_x.long().to(self.device)
                train_y = train_y.long().to(self.device)
                preds = self(train_x)
                loss_batch = loss_fn(preds, train_y)

                opt.zero_grad()
                loss_batch.backward()
                opt.step()
                loss_epoch += loss_batch.item()

            loss_epoch = loss_epoch / len(train_loader)
            loss_epoch_list.append(loss_epoch)

            self.eval()
            with torch.no_grad():
                train_accuracy, train_recall, train_precision, train_f1, train_confusion_matrix = self.metrics_classification(train_loader)
                test_accuracy, test_recall, test_precision, test_f1, test_confusion_matrix= self.metrics_classification(test_loader)
                train_accuracy_list.append(train_accuracy)
                test_accuracy_list.append(test_accuracy)
                test_recall_list.append(test_recall)
                test_precision_list.append(test_precision)
                test_F1_list.append(test_f1)

            print(f'Epoch:{epoch+1}/{epochs} | Loss:{loss_epoch:.4f} | Train_acc:{train_accuracy*100:.4f}% | Test_acc:{test_accuracy*100:.4f}% | Test_recall:{test_recall*100:.4f}% | Test_pre:{test_precision*100:.4f}% | Test_F1:{test_f1:.4f}' )
        

        # Accuracy
        acc_last = test_accuracy_list[-1]
        print(f'Avg accuracy: {acc_last:.4f}')
        logging.info(f'Avg accuracy: {acc_last:.4f}')

        # Recall
        recall_last = test_recall_list[-1]
        print(f'Avg recall (last 10 epochs): {recall_last:.4f}')
        logging.info(f'Avg recall (last 10 epochs): {recall_last:.4f}')

        # Precision
        precision_last = test_precision_list[-1]
        print(f'Avg precision: {precision_last:.4f}')
        logging.info(f'Avg precision (last 10 epochs): {precision_last:.4f}')

        # F1
        F1_last = test_F1_list[-1]
        print(f'Avg F1: {F1_last:.4f}')
        logging.info(f'Avg F1: {F1_last:.4f}')
                
        plot_accuracy_curve(train_accuracy_list, test_accuracy_list)
        plot_confusion_matrix(test_confusion_matrix)
        plot_loss_curve(loss_epoch_list)


    @staticmethod
    def give_num(num=1024, l=32, m=32):
        a = np.array([x for x in range(num)])
        a = a.reshape((l,m))
        a = np.transpose(a,(1,0))
        a = a.reshape([-1,])
        return a

    @staticmethod
    def Generator_matrix(size, new_size, l=85, m=85):
        num_list = EAPCR.give_num(size, l, m)
        num_list = num_list[num_list < new_size]
        M = np.zeros((new_size, new_size))
        for i in range(len(num_list)):
            num = num_list[i]
            M[i][num] = 1
        return M


    def metrics_classification(self, data_loader):
        self.eval()
        all_preds = []
        all_labels = []
        
        for x_test, y_test in data_loader:
            x_test = x_test.long().to(self.device)
            y_test = y_test.long().to(self.device)
            with torch.no_grad():
                outputs = self(x_test)
                probas = outputs.softmax(dim=1)
                top_prob, top_class = torch.max(probas, 1)
                all_preds.extend(top_class.cpu().numpy())
                all_labels.extend(y_test.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        if C.Positive_class == 'macro':
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        elif C.Positive_class == 'weighted':
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        else:
            recall = recall_score(all_labels, all_preds, average='binary', pos_label=C.Positive_class, zero_division=0)
            precision = precision_score(all_labels, all_preds, average='binary', pos_label=C.Positive_class, zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='binary', pos_label=C.Positive_class, zero_division=0)
        return accuracy, recall, precision, f1, conf_matrix