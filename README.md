
# From BERT to Mamba: Evaluating Deep Learning for Efficient QA Systems

This repository contains the implementation and results of our project, **"From BERT to Mamba: Evaluating Deep Learning for Efficient QA Systems"**, which explores the trade-offs between accuracy and computational efficiency in question-answering (QA) systems. Our work compares several deep learning models, including BERT, T5, LSTM, and Mamba, focusing on their performance on the SQuAD 2.0 dataset.

## Table of Contents
- [Introduction](#introduction)
- [Models Evaluated](#models-evaluated)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)

---

## Introduction

Question Answering (QA) and Machine Reading Comprehension are fundamental NLP tasks that involve interpreting user queries and extracting relevant information from text. Our work compares the following models:

1. **BERT**: A transformer-based model with state-of-the-art performance but high computational costs.
2. **T5**: A versatile text-to-text transformer model.
3. **LSTM**: A traditional RNN-based approach, incorporating attention mechanisms for improved context understanding.
4. **Mamba**: A state-space model (SSM) designed for efficient processing of long sequences.

Through rigorous evaluation, we analyze the trade-offs in accuracy and computational efficiency, focusing on metrics such as Exact Match (EM) scores.

---

## Models Evaluated

### BERT Model
- **Architecture**: Transformer-based with bidirectional attention.
- **Pretraining/Fine-tuning**: Fine-tuned on the SQuAD 2.0 dataset.
- **Evaluation**: Demonstrates high accuracy but requires significant memory.

### T5 Model
- **Architecture**: Text-to-text transformer model.
- **Pretraining/Fine-tuning**: Fine-tuned for QA tasks with structured input and output.
- **Evaluation**: Balances accuracy and versatility, with longer training times.

### LSTM-Based Model
- **Architecture**: RNN with bidirectional LSTM layers and attention mechanisms.
- **Pretraining/Fine-tuning**: Trained on a subset of SQuAD 2.0 due to resource constraints.
- **Evaluation**: Low accuracy compared to transformer-based models.

### Mamba Model
- **Architecture**: State-space model (SSM) with linear scaling, optimized for long sequences.
- **Pretraining/Fine-tuning**: Fine-tuned for QA with synthetic negative examples for unanswerable questions.
- **Evaluation**: Efficient but limited in semantic reasoning.

---

## Dataset

We use the [**SQuAD 2.0**](https://rajpurkar.github.io/SQuAD-explorer/) dataset, a benchmark in QA tasks. It includes:
- 100,000 answerable questions with associated context.
- 50,000 unanswerable questions to test the model's ability to identify insufficient information.

---

## Evaluation Metrics

We use the **Exact Match (EM)** score, a strict metric that calculates the percentage of predictions that exactly match the ground-truth answers, including punctuation and whitespace.

---

## Results

| Model   | Exact Match Score |
|---------|-------------------|
| BERT    | 0.64             |
| T5      | 0.68             |
| LSTM    | 0.093            |
| Mamba   | 0.12             |

BERT and T5 outperform other models, while Mamba struggles with semantic reasoning, highlighting the limitations of its architecture for QA tasks.

---

## Future Work

- **BERT**: Extend training with advanced learning rate schedules and incorporate into ensemble models.
- **T5**: Optimize hyperparameters, apply regularization, and explore lightweight techniques like LoRA and pruning.
- **LSTM**: Enhance architecture with pre-trained embeddings and self-attention mechanisms.
- **Mamba**: Experiment with task-specific pretraining and integrate self-attention for improved context modeling.
