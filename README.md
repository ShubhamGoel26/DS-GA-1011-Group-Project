# From BERT to Mamba: Evaluating Deep Learning for Efficient QA Systems

This project investigates the trade-offs between accuracy and computational efficiency in deep learning models for Question Answering (QA) tasks. We evaluate models such as BERT, T5, LSTM, and Mamba on their ability to deliver precise answers while balancing computational resource requirements. Our findings provide insights into optimizing QA systems for real-world applications.

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Models Evaluated](#models-evaluated)
  - [BERT](#bert)
  - [T5](#t5)
  - [LSTM](#lstm)
  - [Mamba](#mamba)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)

---

## Abstract

This project compares multiple models for QA tasks, focusing on balancing accuracy with computational efficiency. We fine-tune and evaluate models like BERT, T5, and Mamba using Exact Match scores and resource usage, offering insights for practical deployment.

## Introduction

Modern QA tasks demand models capable of interpreting queries and extracting precise answers from large contexts. While transformer-based models like BERT and T5 have achieved state-of-the-art results, their computational costs often limit their scalability. This project explores efficient alternatives such as Mamba, alongside traditional LSTM-based models, comparing their trade-offs between performance and efficiency.

## Models Evaluated

### BERT
- **Variant Used:** `bert-base-uncased` from Hugging Face.
- **Key Features:** Transformer-based, bidirectional attention.
- **Fine-Tuning Details:** Pretrained on the SQuAD dataset with a focus on optimizing answer span prediction.

### T5
- **Variant Used:** `T5-base` from Hugging Face.
- **Key Features:** Text-to-text framework, flexible for various NLP tasks.
- **Fine-Tuning Details:** Fine-tuned on SQuAD with specialized tokenizers and performance tracking.

### LSTM
- **Key Features:** Sequential processing with attention mechanisms.
- **Architecture:** Embedding layers, bidirectional LSTMs, and dense layers for answer span prediction.
- **Challenges:** High training time and limited generalization on complex datasets.

### Mamba
- **Variant Used:** `state-spaces/mamba-130m`.
- **Key Features:** Efficient handling of long sequences with linear scaling.
- **Fine-Tuning Details:** Fine-tuned on SQuAD v2.0 using an adapted script for training and evaluation.

## Dataset

We use the **SQuAD 2.0** dataset, a benchmark for QA tasks. It includes:
- Over 100k answerable questions.
- 50k unanswerable questions requiring models to differentiate between answerable and unanswerable queries.

The dataset can be downloaded from the [SQuAD Explorer](https://rajpurkar.github.io/SQuAD-explorer/).

## Evaluation

- **Metric:** Exact Match (EM) score, assessing the percentage of predicted answers that exactly match ground-truth answers.
- **Setup:** Models were trained and evaluated under standardized conditions for fair comparison.

## Results

| Model   | Exact Match Score |
|---------|-------------------|
| BERT    | 0.64             |
| T5      | 0.68             |
| LSTM    | 0.093            |
| Mamba   | 0.12             |

## Future Work

- **BERT and T5:** Enhance performance through hyperparameter tuning, additional epochs, and regularization techniques.
- **LSTM:** Improve generalization with pre-trained embeddings and advanced attention mechanisms.
- **Mamba:** Test on diverse datasets like Natural Questions or TriviaQA to refine its span-based QA capabilities.
