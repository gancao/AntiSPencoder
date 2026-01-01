# AntiSPencoder
Embed TCR CDR3 and antigen peptide amino acid sequences into low-dimensonal, continuous and numerical vectors 

![Figure](https://github.com/gancao/AntiSPencoder/blob/main/AntiSPencoder.png)




## Overview

AntiSPencoder is a BERT-based deep learning model that encodes TCR CDR3 and antigen peptide sequences into low-dimensional, meaningful embeddings. This approach enables a systematic investigation of TCR specificity across diverse antigens, healthy tissues, and cancer types. Through extensive validations, we demonstrate that AntiSPencoder effectively clusters TCRs with similar CDR3 sequences and antigen-specificities in the embedding space, providing a robust computational foundation for subsequent analyses. AntiSPencoder is a python package written in Python 3.9 and pytorch 1.12.


## Installation

[1] Install <a href="https://www.anaconda.com/" target="_blank">Anaconda</a> if not already available

[2] Clone this repository:
```
    git clone https://github.com/gancao/AntiSPencoder.git
```

[3] Change to AntiSPencoder directory:
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

[6] Install AntiSPencoder:
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

### Run cases

See <a href="https://github.com/gancao/AntiSPencoder/blob/main/AntiSP_analysis.py" target="_blank">Cases</a> for performing embeddings.

### Quick start
```
    model_dir = "../model"

    if torch.cuda.is_available():
        device = torch.device("cuda")  
    else:
        device = torch.device("cpu")
    
    encoder_path = os.path.join(model_dir, "filter_C_TCRdb_AntiSPencoder_checkpoint.ep29")
    pretrain = AntiSPencoder.TCREPbert(device=device)

    state_dict = torch.load(encoder_path, map_location=device)['model_state_dict'] # 加载文件
    pretrain.load_state_dict(state_dict)
    pretrain = pretrain.to(device)
    pretrain.eval()
    for param in pretrain.parameters():
        param.requires_grad = False
    pretrain.eval()

    sequence = ['CAAPQAGTALIF','CTDLNTGGFKTIF','CAGPTGGSYIPTF','CAMHRDDKIIF']
    encoder_info,embeddings_info = pretrain.predict(sequence,device=device,batch_size=4,num_workers=4)
```