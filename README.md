# AntiSPencoder
Embed TCR CDR3 and antigen peptide amino acid sequences into low-dimensonal, continuous and numerical vectors 

![流程图](https://github.com/user-attachments/assets/ab27b716-19fa-4172-baad-e2a04958625e)




## Overview

RedeVHD is an efficient spatial transcriptomics deconvolution algorithm designed for high-definition ST data. By directly modeling the relationship between bin-level gene expression and cell-type abundance, the algorithm can process datasets comprising millions of bins or spots within tens of minutes with GPU acceleration. RedeVHD is a python package written in Python 3.9 and pytorch 1.12.


## Installation

[1] Install <a href="https://www.anaconda.com/" target="_blank">Anaconda</a> if not already available

[2] Clone this repository:
```
    git clone https://github.com/Roshan1992/RedeVHD.git
```

[3] Change to RedeVHD directory:
```
    cd RedeVHD
```

[4] Create a conda environment with the required dependencies:
```
    conda env create -f environment.yml
```

[5] Activate the RedeVHD_env environment you just created:
```
    conda activate RedeVHD_env
```

[6] Install RedeVHD:
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


### Run RedeVHD

See <a href="https://github.com/Roshan1992/VHD/blob/main/example.ipynb" target="_blank">example</a> for implementing VHD.

