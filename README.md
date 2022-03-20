# RBP-TSTL is a two-stage transfer learning framework for genome-scale prediction of RNA-binding proteins
## Introduction

RNA-binding proteins (RBPs) form ribonucleoprotein complexes when binding with RNAs. They play crucial roles in the post-transcriptional regulation of RNAs and are potential biological markers for cancer diagnosis. The wet-lab experimental methods, such as RNA Interactome Capture (RIC), can accurately identify RBPs. However, these approaches are time-consuming and cost-intensive, which are not suitable for high-throughput identification of RBPs. In addition, RIC can only work on RBPs with functional poly(A) tails on transcripts and efficient incorporation of cross-link-enhancing artificial nucleotides, thereby limiting its application to RBPs in prokaryotes. In this context, computational approaches for in silico high-throughput prediction of RBPs can help guide hypothesis-driven experimental studies of RBPs. To date, a variety of computational methods have been developed for this purpose. These methods can be generally classified into two major groups: template-based and machine learning-based.
Framework for RBP-TSTL:
![figure_1_v4](https://user-images.githubusercontent.com/72983482/154390499-45bed4dc-a8f4-4c38-9a5a-e91b0d112842.png)

## Steps for re-training the model for genome-scale prediction of RBPs:
* Download the [embeddings](https://drive.google.com/drive/folders/18nCJNhuC_v0GuvxgztJ6YDPKq3wH3YhD?usp=sharing) generated by [Prot-T5](https://github.com/agemagician/ProtTrans) and [the RBPs datasets](https://drive.google.com/drive/folders/1T_PGS3Ake0HGG7S4A4nopcJY5ieQur_4?usp=sharing).
- Run the code 
    - Initialize the customized deep learning model on the annotated pre-training dataset
    ```
    python train.py     --pro_label_dir 'pretrain_accending_trP2392_trN38582_VaP292_VaN4881_TeP298_TeN4889_pep_label.csv'   \ 
                        --rep_dir     'prot_t5_xl_uniref50_pretrain.csv'    \
                        --batch_size  2048       \
                        --epoch 250 
    ```
    - Fine-tune the customized deep learning model on the dataset of target species (Taking *Homo Sapiens* as example)
    
    ```
    python train.py     --pro_label_dir '9606_accending_trP1170_trN8485_VaP126_VaN942_TeP178_TeN1202_pep_label.csv'   \ 
                        --rep_dir     'prot_t5_xl_uniref50_9606.csv'    \
                        --batch_size  2048       \
                        --load_model_dir pretrained_model.pl   \
                        --epoch 250 
    ```
- Alternatively, the genomic prediction results of RBPs on 4 species can be downloaded directly from [google drive](https://drive.google.com/file/d/1i0oYZCt24j5--rBCuUeky81qEOZwlLT9/view?usp=sharing).

## Reference
* Peng,X. et. al., RBP-TSTL is a two-stage transfer learning framework for genome-scale prediction of RNA-binding proteins, submitted for publication
