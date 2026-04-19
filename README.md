# BiLSTM-Attention-CRF from Scratch for Chinese Word Segmentation

This repository contains a high-performance, fully batched implementation of the **BiLSTM-Attention-CRF** model for **Chinese Word Segmentation (CWS)**, built entirely from scratch using **PyTorch**.

Unlike standard implementations that rely on external sequence-labeling libraries, this project focuses on the **low-level mathematical realization** and **engineering optimization** of the CRF layer and Transformer-based attention mechanisms. By leveraging 3D tensor broadcasting and dynamic residual scaling, the model achieves highly efficient parallel computation and robust context modeling.

## Key Features

* **Multi-Head Attention Integration**: Fused the traditional BiLSTM architecture with an 8-head `nn.MultiheadAttention` layer to capture long-range global dependencies without losing local context sensitivity.
* **Learnable Residual Scaling**: Introduced a learnable scalar weight to dynamically balance local structural features (LSTM) and global semantic features (Attention), effectively mitigating overfitting and noise injection on smaller corpora.
* **Fully Batched Zero-API CRF**: The core CRF logic (Forward Algorithm and Viterbi Decoding) is implemented via manual tensor operations, utilizing masking and parallelized log-sum-exp calculations to bypass the performance bottlenecks of traditional batch loops.
* **Hardware-Accelerated Engineering**: Incorporates Automatic Mixed Precision (AMP) training via `torch.amp` and dynamic sequence truncation, maximizing GPU utilization while preventing Out-Of-Memory (OOM) failures caused by outlier sequence lengths.
* **Advanced Training Regimes**: Features Dropout, L2 Regularization, and Dynamic Learning Rate Scheduling (ReduceLROnPlateau) to ensure stable convergence.

## Dataset & Data Source

The model was trained and evaluated on the **SIGHAN PKU (Peking University)** benchmark dataset, ensuring strict alignment between training and testing corpus standards.

### Data Details:
* **Source**: The Second International Chinese Word Segmentation Bakeoff (SIGHAN 2005).
* **Annotation Scheme**: The project utilizes the **BMES (4-tag)** system:
    * **B**: Beginning of a word.
    * **M**: Middle of a word.
    * **E**: End of a word.
    * **S**: Single character word.
* **Preprocessing**: Features custom character-to-index mapping and dynamic sequence padding with mask generation for variable-length batches.

## Performance

After rigorous training and architectural optimization on the SIGHAN PKU dataset, the model achieved the following state-of-the-art level performance for a scratch-built, non-pretrained system:

| Metric | Score |
| :--- | :--- |
| **Precision** | **92.4%** |
| **Recall** | **92.0%** |
| **F1 Score** | **92.2%** |
| **OOV Recall Rate** | **92.0%** |
| **IV Recall Rate** | **91.9%** |

*Note: The model demonstrates exceptional generalization capabilities, maintaining a 92.0% recall rate on Out-Of-Vocabulary (OOV) words despite a highly challenging OOV rate within the test set.*

## Tech Stack & Requirements

* **Language**: Python
* **Framework**: **PyTorch**
* **Core Methodology**: 
    * Bi-directional LSTM for local feature extraction.
    * Multi-Head Self-Attention for global context modeling.
    * Conditional Random Fields (CRF) for sequence-level label optimization.
    * Automatic Mixed Precision (AMP) and 3D Tensor Broadcasting.

## Research Context

This project was developed as a technical deep-dive into NLP architectures and low-level matrix optimization.

## License

Distributed under the MIT License.
