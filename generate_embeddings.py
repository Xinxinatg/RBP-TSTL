import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import gc
import sys
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()
# sequences_Example = ["AETCZAO","S K T Z P"]
inFile = open(sys.argv[1],'r')

headerList = []
seqList = []
currentSeq = ''
for line in inFile:
   if line[0] == ">":
      headerList.append(line[1:].strip())
      if currentSeq != '':
         seqList.append(currentSeq)

      currentSeq = ''
   else:
      currentSeq += line.strip()

seqList.append(currentSeq)
sequences_Example=seqList
strs_1 = ["".join(seq.split()) for seq in sequences_Example]
strs_2 = [re.sub(r"[UZOB]", "X", seq) for seq in strs_1]      
seqs = [ list(seq) for seq in strs_2]
ids=tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")

input_ids = ids['input_ids']
# print('ids.shape',input_ids.shape)
attention_mask = ids['attention_mask']
input_ids,attention_mask=input_ids.to(device),attention_mask.to(device)
with torch.no_grad():
     embedding =  model(input_ids=input_ids,attention_mask=attention_mask)[0] 
features = [] 
for seq_num in range(len(embedding)):
    seq_len = (attention_mask[seq_num] == 1).sum()
    seq_emd = embedding[seq_num][:seq_len-1]
    features.append(torch.mean(seq_emd.cpu(),axis=0).numpy())

import pandas as pd
ft_pd=pd.DataFrame(features)
ft_pd.to_csv('features_mean.csv',index=False)
