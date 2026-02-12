# AntiSPencoder
Embed TCR CDR3 and antigen peptide amino acid sequences into low-dimensonal, continuous and numerical vectors 

![Figure](https://github.com/gancao/AntiSPencoder/blob/main/AntiSPencoder.png)




## Overview

AntiSPencoder is a BERT-based deep learning model that encodes TCR CDR3 and antigen peptide sequences into low-dimensional, meaningful embeddings. This approach enables a systematic investigation of TCR specificity across diverse antigens, healthy tissues, and cancer types. Through extensive validations, we demonstrate that AntiSPencoder effectively clusters TCRs with similar CDR3 sequences and antigen-specificities in the embedding space, providing a robust computational foundation for subsequent analyses. AntiSPencoder is a python package written in Python 3.9 and pytorch 1.12.

Execept for TCR embeddings, we developed R programs for predicting potential TCR antigen-specificity via known TCR-pMHC interactions. Additionally, we also provided a user-friendly interactive HTML result page for users to browse the predicted results for potential TCR antigen-specificity in various graphs and data tables.


## Installation

[1] Install <a href="https://www.anaconda.com/" target="_blank">Anaconda</a> if not already available

[2] Clone this repository:
```
    git clone https://github.com/gancao/AntiSPencoder.git
```

[3] Change to AntiSPencoder directory including setup.py:
```
    cd AntiSPencoder
```

[4] Create a conda environment with the required dependencies if you need:
```
    conda env create -f environment.yml
```

[5] Activate the AntiSPencoder_env environment if you created:
```
    conda activate AntiSPencoder_env
```

[6] You can also directly install AntiSPencoder:
```
    pip install .
```

[7] Install pytorch:

If GPU available (https://pytorch.org/get-started/previous-versions/):
```
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
If GPU not available:
```
    pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run AntiSPencoder

### Quick start

The python codes:
```
    model_dir = "model"
    if torch.cuda.is_available():
        device = torch.device("cuda")  
    else:
        device = torch.device("cpu")
    
    encoder_path = os.path.join(model_dir, "filter_C_TCRdb_AntiSPencoder_checkpoint.ep29")
    pretrain = AntiSPencoder.TCREPbert(device=device)

    state_dict = torch.load(encoder_path, map_location=device)['model_state_dict']
    pretrain.load_state_dict(state_dict)
    pretrain = pretrain.to(device)
    pretrain.eval()
    for param in pretrain.parameters():
        param.requires_grad = False
    pretrain.eval()

    cdr3_file = "data/celltypist/TILC_TCR_onlyseq_simple_metadata.txt"
    cdr3_info = pd.read_csv(cdr3_file,header=0,index_col=0,sep="\t")
    sequence = list(set(cdr3_info['cdr3'].dropna().tolist())
    encoder_info,embeddings_info = pretrain.predict(sequence,device=device,batch_size=4,num_workers=4)
    encoder_info.to_csv("analysis/celltypist/encoder_info.txt",sep="\t")
    embeddings_info.to_csv("analysis/celltypist/embeddings_info.txt",sep="\t")
```
### Run cases

See <a href="https://github.com/gancao/AntiSPencoder/blob/main/script/AntiSP_analysis.py" target="_blank">Cases</a> for performing embeddings in this paper.

## Prediction of TCR antigen-specificity based on known TCR-epitope pairs

The R script predict_epitope_code.R require three key data files deposited on zenodo website. More details can been seen from https://zenodo.org/records/18113375

### Usage

``` 
    #Install required R packages
    Rscript install_r_packages.R
```
```
    #load R functions
    source("script/predict_epitope_code.R",chdir = T)
    #current path:script

    #load data
    cdr3_epitope_emb_file <- "../data/TCR_epitope/tcr_epitope_complete_cdr3_filter_embedding_info.txt"
    epitope_cutoff_file <- "../data/TCR_epitope/AntiSPencoder_test_final_dist_cutoff_results_seed1996.txt"
    cdr3_epitope_path <- "../data/TCR_epitope/merge_TCR_epitope_data_complete_cdr3_filter_rename_antigen_final.txt"
    cdr3_epitope_info <- read.delim(cdr3_epitope_path,header=TRUE,row.names=1,check.names=FALSE)
    cdr3_epitope_info <- subset(cdr3_epitope_info,select = -data_source)
    cdr3_epitope_info <- subset(cdr3_epitope_info,select = -cell_type)
    cdr3_epitope_info <- cdr3_epitope_info[!duplicated(cdr3_epitope_info),]
    cdr3_epitope_info$ID <- rownames(cdr3_epitope_info)
    cat(nrow(cdr3_epitope_info)," records")
    cdr3_epitope_emb_info <- read.delim(cdr3_epitope_emb_file, header=TRUE, check.names=FALSE, row.names=1)
    epitope_cutoff_info <- read.delim(epitope_cutoff_file,header=TRUE)
    rownames(epitope_cutoff_info) <- as.character(epitope_cutoff_info$epitope)

    #predictions
    cdr3_file <- "../data/celltypist/TILC_TCR_onlyseq_simple_metadata.txt"
    cdr3_info = read.delim(cdr3_pair_file,header = TRUE,row.names=1,check.names = F)
    chains_info = as.character(cdr3_info$chain)
    names(chains_info) <- as.character(cdr3_info$cdr3)

    embeddings_info = read.delim("../analysis/celltypist/embeddings_info.txt",header=T,check.names=F)
    rownames(embeddings_info) <- as.character(embeddings_info$sequence)
    embeddings_info = embeddings_info[,-(1:2)]
    chain_info = chains_info[rownames(embeddings_info)]
    pred_epitopes = predict_epitope_by_min_distance(embeddings_info,chain_info,cores=4,aa_dist=3)

    #visualizations 
    pred_pred_epitopes_file <- "../analysis/celltypist/AntiSPencoder_celltypist_pred_epitopes_info_by_min_dist.txt"
    pred_epitopes <- read.delim(pred_pred_epitopes_file,header=TRUE,check.names=F)
    
    #you can filtered the results for visualization by edit distance:
    pred_epitopes <- pred_epitopes[pred_epitopes$lv_distance<=3,,drop=FALSE]

    #You can also set your CDR3 names for visualizations,or = NULL
    rm_pred_cdr3_info <- pred_epitopes[!duplicated(pred_epitopes[,c("pred_cdr3","pred_chain")]),]
    metadata_info <- paste(rm_pred_cdr3_info$pred_chain,1:nrow(rm_pred_cdr3_info))
    names(metadata_info) <- as.character(rm_pred_cdr3_info$pred_cdr3)
    
    cdr3_pair_file <- "../data/celltypist/tcr_pair_info.txt"
    cdr3_pair_info = read.delim(cdr3_pair_file,header = TRUE,row.names=1,check.names = F)

    visualize_pred_info(pred_epitopes,result_dir="../analysis/celltypist",prefix="celltypist",metadata_info=metadata_info,cdr3_pair_info=cdr3_pair_info,ncores=10,plot_MSA=TRUE)
```