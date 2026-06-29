# U-Net Semantic Segmentation

A PyTorch implementation of semantic image segmentation on the Oxford-IIIT Pet dataset. This project includes both a custom U-Net implementation and a pretrained U-Net (ResNet-18 encoder) for comparison. Training and evaluation are tracked using Weights & Biases.

## Features

* Custom U-Net implemented from scratch
* Pretrained U-Net using `segmentation_models_pytorch`
* Oxford-IIIT Pet semantic segmentation
* AdamW optimizer with CosineAnnealingLR
* Mean IoU and Dice evaluation metrics
* Weights & Biases experiment tracking

## Repository Structure

```text
data/
outputs/
src/
├── dataset.py
├── metrics.py
├── model.py
├── train.py
tests/
├── test_dataset.py
├── test_metrics.py
├── test_model.py
README.MD
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

In the root folder run:

```bash
python -m src.train
```

Model type (custom or pretrained) can be selected by changing the `pretrained` flag inside `train.py`.

## Results

Training logs include:

* Training and validation loss
* Learning rate schedule
* Mean IoU
* Mean Dice coefficient
* Predicted segmentation masks

A detailed experiment report is available on Weights & Biases.


## 📊 Experiment Report

📄 **Weights & Biases Report:** [Semantic Segmentation of Oxford-IIIT](https://wandb.ai/bmj-university-of-reiburg/u-net_semantic_segmentation/reports/Semantic-Segmentation-of-Oxford-IIIT--VmlldzoxNzM2Njg4MQ)
