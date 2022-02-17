# RBP-TSTL
\backslash
Framework for RBP-TSTL:
![figure_1_v4](https://user-images.githubusercontent.com/72983482/154390499-45bed4dc-a8f4-4c38-9a5a-e91b0d112842.png)

# The genomic prediction results of RBPs on 4 species can be downloaded from following link:
https://drive.google.com/file/d/1i0oYZCt24j5--rBCuUeky81qEOZwlLT9/view?usp=sharing

# Steps for implementing the code:
1. Download the embeddings generated by self-supervised pre-trained language model and the RBPs datasets from https://drive.google.com/drive/folders/18nCJNhuC_v0GuvxgztJ6YDPKq3wH3YhD?usp=sharing, https://drive.google.com/drive/folders/1T_PGS3Ake0HGG7S4A4nopcJY5ieQur_4?usp=sharing
2. run the code 
!python train.py     --pro_label_dir 'pretrain_accending_trP2392_trN38582_VaP292_VaN4881_TeP298_TeN4889_pep_label.csv'          $\$
                     --rep_dir     'prot_t5_xl_uniref50_pretrain.csv'     
                     --batch_size  2048   \
                     --epoch 250\
