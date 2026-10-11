# Simple Deep Belief Network for Pattern Recognition

**Course:** Deep Learning  
**Project:** Implement a small Deep Belief Network (DBN) to learn hierarchical representations from an image/pattern dataset and perform classification.

## Overview

This project demonstrates a DBN-style learning pipeline on the MNIST handwritten-digit dataset. It greedily pretrains two Bernoulli Restricted Boltzmann Machines (RBMs) using Contrastive Divergence (CD-k), stacks their hidden representations, and trains a supervised linear classifier on the learned features.

## Features

- Stacked RBMs and greedy layer-wise unsupervised pretraining
- MNIST digit classification (0–9)
- Test accuracy, precision/recall/F1 report, and confusion matrix
- Visualizations of learned first-layer weights and sample predictions
- Streamlit interface for uploading an image and viewing a predicted digit

## Repository structure

```text
dbn-pattern-recognition/
├── README.md
├── requirements.txt
├── app.py
├── dataset/                 # MNIST downloads here automatically
├── src/
│   └── dbn_pattern_recognition.py
├── notebooks/
│   └── DBN_Analysis.ipynb   # optional analysis notebook
├── models/                  # trained checkpoint saved here
├── results/                 # metrics and plots saved here
└── screenshots/             # add screenshots of your own app output
```

## Setup

Python 3.10 or 3.11 is recommended. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python src/dbn_pattern_recognition.py
```

The first run downloads MNIST. The default configuration trains both RBMs for 3 epochs and the classifier for 5 epochs. Full training may take a while on a CPU.

For a quick smoke test, run fewer batches and epochs:

```bash
python src/dbn_pattern_recognition.py --rbm-epochs 1 --classifier-epochs 1 --limit-batches 20
```

The quick test is only for checking that the pipeline runs; its accuracy is not a final result. After training, the model is saved to `models/dbn_mnist.pt` and plots/metrics are saved to `results/`.

## Launch the interactive application

After training:

```bash
streamlit run app.py
```

Upload a centered, high-contrast handwritten digit image. The app shows the predicted digit and probabilities for digits 0–9.

## Methodology

1. Load MNIST and scale grayscale pixel values to [0, 1].
2. Flatten each 28 × 28 image into a 784-dimensional vector.
3. Pretrain RBM 1 on pixels using Contrastive Divergence.
4. Use RBM 1 hidden probabilities as input to RBM 2 and pretrain it.
5. Pass images through the stacked RBMs to obtain learned feature representations.
6. Train a linear classification head with cross-entropy loss using labeled MNIST data.
7. Evaluate on the test set and save metrics and visualizations.

## Important interpretation

This is an educational, compact DBN implementation. The RBMs are trained layer-by-layer, and the classifier is trained on the resulting frozen features. It is not a CNN. Results depend on epochs, hyperparameters, random initialization, and compute hardware. Record your actual output metrics after running the code; do not report assumed accuracy.

## Academic integrity

Understand and customize the code yourself. Make meaningful commits as you develop, cite external references you use, and be ready to explain RBMs, Contrastive Divergence, greedy pretraining, classification, and evaluation during the viva.

## References

- Hinton, G. E. (2009). *A Practical Guide to Training Restricted Boltzmann Machines*.
- LeCun, Y., Cortes, C., & Burges, C. J. C. *The MNIST Database of Handwritten Digits*.
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- Torchvision datasets documentation: https://pytorch.org/vision/stable/datasets.html
- Streamlit documentation: https://docs.streamlit.io/
