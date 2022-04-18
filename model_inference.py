import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
import pandas as pd
# from termcolor import colored
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_curve,auc,roc_auc_score
import os
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score, matthews_corrcoef
import math
from tqdm.notebook import tqdm
import sys
parser = argparse.ArgumentParser(description='embeddings_for_RBP_prediction')
parser.add_argument('--species', type='9606', default=200, help='epoch number')
parser.add_argument('--model_dir', default='Model/', help='model directory')
parser.add_argument('--rep_dir', help='represention file directory')
args = parser.parse_args()
cl_dict={'9606':0.61,'3701':0.42,'590':0.50,'561':0.65}
class newModel1(nn.Module):
    def __init__(self, vocab_size=26):
        super().__init__()
        self.hidden_dim = 256
        self.batch_size = 256
        self.emb_dim =1024
        
        # self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.gmlp_t=gMLP(num_tokens = 1000,dim = 32, depth = 2,  seq_len = 40, act = nn.Tanh())
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=6, 
                               bidirectional=True, dropout=0.05)
        
        
        self.block1=nn.Sequential(nn.Linear(3584,1024),
                                            nn.BatchNorm1d(1024),
                                            nn.LeakyReLU(),
                                            nn.Linear(1024,512),
                                            nn.BatchNorm1d(512),
                                            nn.LeakyReLU(),
                                            nn.Linear(512,256),
                                 )

        self.block2=nn.Sequential(
                                               nn.BatchNorm1d(256),
                                               nn.LeakyReLU(),
                                               nn.Linear(256,128),
                                               nn.BatchNorm1d(128),
                                               nn.LeakyReLU(),
                                               nn.Linear(128,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,2)
                                            )
        
    def forward(self, x):
        # x=self.embedding(x)
        # output=self.transformer_encoder(x).permute(1, 0, 2)
        # output=self.gmlp_t(x).permute(1, 0, 2)
        x=x.view(1,x.shape[0],x.shape[1])
        # output=self.gmlp_t(x).permute(1, 0, 2)
        # print(output.shape)
        output,hn=self.gru(x)
        output=output.permute(1,0,2)
        hn=hn.permute(1,0,2)
        output=output.reshape(output.shape[0],-1)
        hn=hn.reshape(output.shape[0],-1)
        output=torch.cat([output,hn],1)
        # print('output.shape',output.shape)
        output=self.block1(output)
        return self.block2(output)
# model_dir='/home/xinxinpeng/jupyter_book/Model/9606_prottrans0.pl'
# model_dir='/home/xinxinpeng/jupyter_book/Model/fine_tune9606_prottrans0.pl'
model_dir=args.model_dir

# jupyter_book/Model/fine_tune9606_prottrans0.pl
rep_dir=args.rep_dir

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
model=torch.load(model_dir)
# net=nn.DataParallel(newModel1()).double().to(device)
net=newModel1().double().to(device)
net.load_state_dict(model['model'])
data=torch.tensor(pd.read_csv(rep_dir).values)
# dataset = Data.TensorDataset(data)
data_iter = torch.utils.data.DataLoader(data, batch_size=2048, shuffle=False)
net.eval()
logits_all=[]
soft_max=nn.Softmax(dim=1)
for rep in tqdm(data_iter):
#     print(len(rep))
#     print(rep.shape)
    rep=rep.to(device)
    with torch.no_grad(): 
        logits=net(rep)
        logits_all.append(logits/0.50)
logits_in_one=torch.vstack(logits_all)
print(soft_max(logits_in_one))

