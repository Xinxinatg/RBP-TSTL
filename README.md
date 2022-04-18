# RBP-TSTL is a two-stage transfer learning framework for genome-scale prediction of RNA-binding proteins
## Introduction

RNA binding proteins (RBPs) are crucial in the post-transcriptional control of RNAs and play vital roles in a myriad of biological processes, such as RNA localization and gene regulation. Therefore, computational methods that are capable of accurately identifying RBPs are highly desirable and have important implications for biomedical and biotechnological applications. Here we propose a two-stage deep transfer learning-based framework, termed RBP-TSTL, for accurate prediction of RBPs. In the first stage, the knowledge from the self-supervised pre-trained model was utilized for feature embeddings to represent the protein sequence, while in the second stage, a customized deep learning model was initialized based on an annotated pre-training RBPs dataset before being fine-tuned on each corresponding target species dataset. This two-stage transfer learning framework can enable the RBP-TSTL model to be effectively trained to learn and improve the prediction performance. Extensive performance comparison between the RBP-TSTL models trained using the features generated by the self-supervised pre-trained model and other models trained using hand-crafting encoding features demonstrated the effectiveness of the proposed two-stage knowledge transfer strategy based on the self-supervised pre-trained models. Using the best-performing RBP-TSTL models, we further conducted genome-scale RBP predictions for *Homo sapiens*, *Arabidopsis thaliana*, *Escherichia coli*, and *Salmonella* and established a computational compendium containing all the predicted putative RBPs candidates. 
![figure_1_v4](https://user-images.githubusercontent.com/72983482/154390499-45bed4dc-a8f4-4c38-9a5a-e91b0d112842.png)
We anticipate that the proposed RBP-TSTL approach will be explored as a useful tool for the characterization of RNA-binding proteins and exploration of their sequence-structure-function relationships.
## Dependency
* python 3.8
* torch 1.7.1
* cuda 11.0
* scikit_learn 0.22.2 
* SentencePiece 
* transformers
## Steps for re-training the model for genome-scale prediction of RBPs:
* Download the [embeddings](https://drive.google.com/drive/folders/17hbcbqwrFsxam3njYmPsUMb_3iycbLHK?usp=sharing) generated by [Prot-T5](https://github.com/agemagician/ProtTrans) and [the RBPs datasets](https://drive.google.com/drive/folders/1T_PGS3Ake0HGG7S4A4nopcJY5ieQur_4?usp=sharing).
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

## Steps for infering potential RBPs on 4 species using trained models:
* Download the [trained models](https://drive.google.com/drive/folders/1ZGBkgefSMD0hxILR9AIF7VG8yWfikg1q?usp=sharing)
- Run the code 
    - Generating embeddings of the protein sequences of the potential RBPs:
    ```
    python generate_embeddings.py [fasta file of the sequences of potential RBPs]
    ```
    - Loading trained model and output confidence level (Taking *Homo Sapiens* as example):
    
    ```
    python model_inference.py       --species '9606'   \ 
                                    --rep_dir     'prot_t5_xl_uniref50_9606.csv'    \
                                    --model_dir '9606_model.pl'   
    ```

## Download of genomic scale prediction results of RBPs on 4 species:
Alternatively, the genomic scale prediction results of RBPs on 4 species can be downloaded directly from [google drive](https://drive.google.com/file/d/1i0oYZCt24j5--rBCuUeky81qEOZwlLT9/view?usp=sharing).
## Reference
* Peng,X. et. al., RBP-TSTL is a two-stage transfer learning framework for genome-scale prediction of RNA-binding proteins, submitted for publication
