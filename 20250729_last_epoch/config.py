import torch
import pandas as pd

data_path = '../data/ID/data_ID.csv'

outpath = '../output_EAPCR'

test_size =0.3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 2025


num_embed = 112
embed_dim = 128
dropout_prob = 0


epochs = 1000
batch_size = 64

learning_rate = 0.00001

# Positive_class = 'macro'
# Positive_class = 'weighted'
Positive_class = 0