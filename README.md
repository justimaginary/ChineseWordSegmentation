# BiLSTM-CRF from Scratch for Chinese Word Segmentation

This repository contains a high-performance implementation of the **BiLSTM-CRF** model for **Chinese Word Segmentation (CWS)**, built entirely from scratch using **PyTorch**.

Unlike standard implementations that rely on external libraries, this project focuses on the **low-level mathematical realization** of the CRF layer. By leveraging **3D tensor broadcasting** and **dimension alignment**, the model achieves efficient parallel computation on GPUs, bypassing the performance bottlenecks of traditional loops.

## Key Features

* **Zero-API CRF Implementation**: The core logic, including the CRF Forward Algorithm and Viterbi Decoding, is implemented via manual tensor operations rather than high-level APIs.
* **Optimized for GPU**: Utilizes vectorized broadcasting to handle batch processing, resulting in a significant increase in inference speed.
* **Advanced Training Regimes**: Incorporates **Dropout**, **L2 Regularization**, and **Dynamic Learning Rate Scheduling** to ensure robust convergence and generalization.
* **Professional Performance**: Achieves great results for a standalone model without the use of pre-trained embeddings.

## Dataset & Data Source

The model was trained and evaluated on the **SIGHAN PKU (Peking University)** benchmark dataset. 

### Data Details:
* **Source**: The Second International Chinese Word Segmentation Bakeoff (SIGHAN 2005).
* **Annotation Scheme**: The project utilizes the **BMES (4-tag)** system:
    * **B**: Beginning of a word.
    * **M**: Middle of a word.
    * **E**: End of a word.
    * **S**: Single character word.
* **Preprocessing**: Features custom character-to-index mapping and dynamic sequence padding with mask generation for variable-length batches.

## Performance

After rigorous training on the SIGHAN PKU dataset, the model achieved the following performance:

| Metric | Score |
| :--- | :--- |
| **F1 Score** | **91.0%** |

*The model successfully learned precise grammatical patterns through deep feature extraction, even with randomly initialized embeddings.*

## Tech Stack & Requirements

* **Language**: Python
* **Framework**: **PyTorch**
* **Methodology**: 
    * Bi-directional LSTM for global context modeling.
    * Conditional Random Fields (CRF) for sequence-level label optimization.
    * 3D Tensor Broadcasting for parallelized log-sum-exp calculations.

## Research Context

This project was developed as a technical deep-dive while pursuing:
* **Tongji University**: Undergraduate research and core CS coursework.

## License

Distributed under the MIT License.

**Author**: Tianhong Xie
**Institution**: Tongji University, Major in Information Security
**Personal Website**: [justimaginary.github.io](https://justimaginary.github.io)
