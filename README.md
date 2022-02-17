# RBP-TSTL
The genomic prediction results of RBPs on 4 species can be downloaded from following link:
https://drive.google.com/file/d/1i0oYZCt24j5--rBCuUeky81qEOZwlLT9/view?usp=sharing

Steps for implementing the code:
1. Download the embeddings generated by self-supervised pre-trained language model and the RBPs datasets from https://drive.google.com/drive/folders/18nCJNhuC_v0GuvxgztJ6YDPKq3wH3YhD?usp=sharing, https://drive.google.com/drive/folders/1T_PGS3Ake0HGG7S4A4nopcJY5ieQur_4?usp=sharing
2. run the code 
   #                               --load_model_dir '/home/xinxinpeng/RBPs_study/ablation_study/Model/pretrain_prottrans0.pl'
!python train.py   --pro_label_dir '/home/xinxinpeng/RBPs_study/all_hand_crafted_feats_comparison/rna_binding_4species_dataset_pre_dataset/pre_dataset/pretrain_accending_trP2392_trN38582_VaP292_VaN4881_TeP298_TeN4889_pep_label.csv'          \
                              --rep_dir     '/home/xinxinpeng/RBPs_study/RBP_dataset_miscilious/embeddings/prottrans/prot_t5_xl_uniref50_pretrain.csv'    \
                              --batch_size  2048 \
                              --epoch 250\
#                               --load_model_dir '/home/xinxinpeng/RBPs_study/ablation_study/Model/pretrain_prottrans0.pl'
