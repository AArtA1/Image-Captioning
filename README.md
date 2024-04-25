# Image Captioning using Transformer

- [1. Introduction](#1-introduction)
- [2. Run](#2-run)
  - [2.1. **Notion**](#21-important-note)
  - [2.2. Requirements](#22-requirements)
  - [2.3. Create Dataset](#23-create-dataset)
  - [2.4. Create and train Custom Tokenizer](#23-create-dataset)
  - [2.4. Train the model](#24-train-the-model)
- [3. The Model](#3-the-model)
  - [3.1. Introduction](#31-introduction)
  - [3.2. Framework](#32-framework)
  - [3.3. Training](#33-training)
- [4. References](#4-references)

## 1. Introduction

This document provides a comprehensive guide on running the code, delves into the intricacies of the model architecture, discusses the training process including hyperparameters and performance metrics.

## 2. Run

### 2.1. Notion

An important note regarding the implementation of the transformer attention mechanism is provided.  The main purpose of this project is to explore all state-of-the-art techniques, make own pipeline which consists from custom_tokenizer, custom attention implementation and  Please refer to this section before proceeding.

### 2.2. Requirements

You also can find requirements.txt file and docker instructions for training and deploying in container.

### 2.3. Create Dataset

In this project was used small dataset Flickr8k(2014) where represented only 8000 examples. Here is given ready pipeline which is implemented in prepare_dataset.py file.

### 2.4  Create and Train Custom Tokenizer

Huggingface provides convenient instruments for training and creating custom tokenizer. The tokenizer training is also presented in prepare_dataset.py.

### 2.5. Train the model

Instructions for training the model, along with the required arguments, are outlined in this section. The training process is tracked using Neptune for experiment management.
To download datase and run it through the pipeline run the docker by calling the commands further
To create image:
```bash
docker build -t your_image_name .
```

To run your image:

```bash
docker run -d your_image_name
```




## 3. The Model

### 3.1. Introduction

This section provides an overview of the transformer-based model utilized for image captioning. It discusses the motivation behind using transformers for this task and provides insights into the model architecture.

### 3.2. Framework

Details of the model framework, including the image patching process, transformer encoder-decoder architecture, and modifications made for this specific project, are discussed in this section.

### 3.3. Training

The training process, including loss function, optimizer, hyperparameters, and evaluation metrics, is explained comprehensively in this section. Neptune is used for experiment tracking.


## 4. References