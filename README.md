# CLIP from Scratch: Vision–Language Alignment with ResNet50 + DistilBERT  

## Overview  

This repository contains a **from-scratch implementation of CLIP (Contrastive Language–Image Pretraining)** using **PyTorch**. Unlike off-the-shelf models, this version is fully built with modularity in mind—leveraging **ResNet-50 as the image encoder** and **DistilBERT as the text encoder**, trained on the **Flickr30k dataset** for cross-modal alignment.  

Our training achieved:  
- **Training Loss**: `7.97e-06`  
- **Validation Loss**: `0.00164`  

This showcases the ability of the model to learn powerful text–image representations with minimal training loss, proving the effectiveness of the architecture and custom loss design.  

## Features  

- 🖼️ **Image Encoder**: ResNet-50 backbone from `timm` for robust visual feature extraction  
- 📝 **Text Encoder**: DistilBERT from Hugging Face Transformers for lightweight yet powerful text embeddings  
- ⚡ **Custom Contrastive Loss**: Modified categorical cross-entropy with softmaxed inputs for stable optimization  
- 📊 **Dataset**: Flickr30k, enabling strong grounding between natural images and captions  
- 🧩 **End-to-End Modular Design**: Easy to swap encoders, losses, and datasets  
- 🚀 **From Scratch**: No reliance on pre-built CLIP models—everything is transparent and customizable  

## Requirements  

Install the dependencies using:  

```bash
pip install torch torchvision timm transformers datasets matplotlib numpy
```

## Libraries Used
torch, torchvision – Deep learning foundation for training and optimization

timm – Pretrained ResNet-50 backbone for image encoding

transformers – Hugging Face API for DistilBERT and tokenization

datasets – Efficient dataset loading and preprocessing (Flickr30k support)

numpy, matplotlib – For visualization and analysis

## Repository Structure
### 1. Data Loading & Preprocessing
Load Flickr30k dataset

Apply tokenization on captions with DistilBERT tokenizer

Apply image transformations compatible with ResNet-50
### 2. Model Architecture
ResNet-50 image encoder (outputs dense embeddings)

DistilBERT text encoder (outputs contextual embeddings)

Projection layers mapping both modalities into a shared embedding space

### 3. Training with Custom Loss
Contrastive learning objective using softmaxed categorical cross-entropy

Optimized for stable text–image similarity learning
### 4. Evaluation
Monitor training and validation losses

Visualize text–image alignment results
