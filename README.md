# Shopformer: Transformer-Based Framework for Detecting Shoplifting via Human Pose
## Overview

This repository contains the official implementation of [Shopformer (CVPR 2025)](https://arxiv.org/abs/2504.19970). Shopformer is a novel transformer-based framework designed for detecting shoplifting behaviors using only human pose data, , rather than raw pixel information. Unlike traditional video-based methods, Shopformer focuses on privacy-preserving, pose-based shoplifting detection. The model introduces a two-stage architecture: A Graph Convolutional Autoencoder (GCAE) learns rich spatio-temporal embeddings from human pose sequences. These embeddings are tokenized and passed through a transformer encoder-decoder, which reconstructs the sequence. The reconstruction error is then used to compute a normality score for shoplifting detection.

## Key Features

- GCAE-based tokenization of human pose sequences
- Transformer encoder-decoder with attention for behavior modeling
- Evaluated on PoseLift dataset (real-world shoplifting pose data)
- Privacy-preserving and real-time capable
  
## Shopformer Architecture
The following figure illustrates the overall architecture of the Shopformer model:
<figure>
  <img src="Images/Shopformer.png" alt="Shopformer Architecture" width="1300"/>
  <figcaption><sub><b>Figure 1:</b> Overview of the Shopformer architecture. The framework operates in two stages: (1) a Graph Convolutional Autoencoder is first trained on pose sequences to learn rich spatio-temporal representations; (2) the pretrained encoder is then repurposed as a tokenizer module, generating compact tokens from input pose data. These tokens are passed through a transformer encoder-decoder module, which reconstructs the input sequence. The reconstruction error (MSE loss) is used to compute the normality score for shoplifting detection.</figcaption>
  </sub></figure>


  ## Project Structure

- `models/` – GCAE tokenizer & transformer model
- `scripts/` – training and evaluation scripts
- `data/` – instructions and expected format for PoseLift dataset
- `config/` – training configurations
- `utils/` – metric calculations, pose preprocessing

## Dataset
This model is trained on the PoseLift dataset. You can access to the dataset nad related documentation here: 
 [PoseLift GitHub Repository](https://github.com/TeCSAR-UNCC/PoseLift)

 After downloading the dataset, organize the files into the following directory structure: 
 
 DATA/
└── Poselift/
    ├── gt/
    │   └── test_frame_mask/
    │       └── (test set frame-level binary mask files indicating normal or anomalous behavior)
    └── pose/
        ├── train/
        │   └── (training pose JSON files)
        └── test/
            └── (test pose JSON files)

## Installation
conda env create -f environment.yml

conda activate shopformer

 ## Training
Stage 1: Train the tokenizer 


Stage 2: Freeze the encoder and train transformer:


## Results
Shopformer generates 2 tokens per pose sequence, as this setup achieved the best trade-off between accuracy and computational efficiency during ablation studies. Each token has an embedding size of 144, encoded using 8 channels over 18 keypoints. For detailed results comparing token counts ranging from 2 to 12, please refer to the ablation study section in the [paper](https://arxiv.org/abs/2504.19970).



<sub> Table 1: AUC-ROC, AUC-PR, and EER of Shopformer compared with state-of-the-art pose-based anomaly detection models on the PoseLift dataset.
| Methods          | AUC-ROC | AUC-PR| EER |
|------------------|---------|-------|-----|
| STG-NF         |    67.46   | 84.06        | 0.39   |
|TSGAD           |   63.35    |  39.31       | 0.41    |
| GEPC          |   60.61    |  50.38       | 0.38  |
|Shopformer    |  69.15  | 44.49 | 0.38 |



## Citation
If you find our work useful, please consider citing: 

```bibetex
@article{rashvand2025shopformer,
  title={Shopformer: Transformer-Based Framework for Detecting Shoplifting via Human Pose},
  author={Rashvand, Narges and Noghre, Ghazal Alinezhad and Pazho, Armin Danesh and Ardabili, Babak Rahimi and Tabkhi, Hamed},
  journal={arXiv preprint arXiv:2504.19970},
  year={2025}
}
```

## Contact
If you have any questions or need assistance, please contact the authors at nrashvan@charlotte.edu.
