from util import CustomDataset, Get_Datas_TXT, Get_Datas_CSV
from model import  EAPCR
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import sys
import logging
import config as C
import os
from datetime import datetime

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__=='__main__':

    seed = C.seed
    set_random_seed(seed)
    

    data_path = C.data_path
    outpath = C.outpath
    
    device = C.device

    epochs = C.epochs
    batch_size = C.batch_size
    learning_rate = C.learning_rate

    num_embed = C.num_embed
    embed_dim = C.embed_dim
    dropout_prob = C.dropout_prob
    
    logpath = os.path.join(outpath, 'log')
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(logpath, exist_ok=True)
    log_file = os.path.join(logpath, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f'seed: {seed}')
    logging.info(f'epochs: {epochs}')
    logging.info(f'batch size: {batch_size}')
    logging.info(f'learning rate: {learning_rate}')
    logging.info(f'dropout probability: {dropout_prob}')
    
    
    if sys.argv[1] == 'train':

        X_train, X_test, y_train, y_test = Get_Datas_CSV(data_path)
        train_dataset = CustomDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = CustomDataset(torch.tensor(X_test),  torch.tensor(y_test))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model = EAPCR(num_embed, embed_dim, dropout_prob, device)
        logging.info("模型结构如下：\n" + str(model))
        model.train_model(train_loader, test_loader, epochs=epochs, lr=learning_rate)