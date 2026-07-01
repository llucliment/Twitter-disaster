# Twitter Disaster Tweet Classification

This repository contains my work on the Kaggle competition [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

The goal of the competition is to predict whether a tweet refers to a real disaster or not. This is a binary NLP classification task where each tweet is labeled as:

- `1` → real disaster
- `0` → not a real disaster

> **Status:** Work in progress.  
> I am actively experimenting with different NLP models, transformer fine-tuning strategies, validation methods, and hyperparameter optimization techniques to improve the final F1 score.

## Project Overview

This project explores different approaches to disaster tweet classification, starting from recurrent neural networks trained from scratch and moving toward transformer-based models such as BERTweet.

The current focus is on fine-tuning BERTweet models using PyTorch and Hugging Face Transformers, with special attention to validation strategy, threshold tuning, and cross-validation.

## Current Approach

The current pipeline includes:

- Tweet preprocessing and input construction
- Integration of the `keyword` feature into the model input
- Fine-tuning BERTweet-base and BERTweet-large
- Training with PyTorch and Hugging Face Transformers
- Early stopping based on validation performance
- F1-score optimization
- Threshold tuning for binary classification
- Experimenting with hyperparameters such as:
  - learning rate
  - batch size
  - max sequence length
  - weight decay
  - warmup ratio
- Exploring stratified K-fold cross-validation
- Testing Optuna-based hyperparameter optimization

## Repository Structure

```text
Twitter-disaster/
├── data/
├── notebooks/
│   ├── rnns_from_scratch.ipynb 
│   ├── bertweet-base.ipynb
│   └── bertweet-large_cv.ipynb
├── src/
│   ├── dataset.py
│   ├── engine.py
│   ├── evaluate.py
│   ├── models.py
│   ├── preprocessing.py
│   └── utils.py
├── submissions/
├── requirements.txt
└── README.md
└── .gitignore