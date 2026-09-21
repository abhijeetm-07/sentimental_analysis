# EmotionAI: NLP Emotion Classification with Deep Sequence Models

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow / Keras](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-2.x-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) and Deep Learning system designed to analyze and classify human emotions in text into six distinct emotional categories: **Joy**, **Sadness**, **Love**, **Anger**, **Fear**, and **Surprise**.

This repository contains the complete experimental pipeline—from raw data extraction and exploratory data analysis (EDA), class imbalance mitigation, baseline ML modeling, sequence modeling (SimpleRNN, LSTM, GRU), to an optimized **Bidirectional GRU (BiGRU)** neural network achieving **~92.25% test accuracy**, paired with a **FastAPI** backend and an interactive web interface.

---

## Table of Contents

- [Overview & Key Features](#overview--key-features)
- [Emotions Classified](#emotions-classified)
- [Dataset](#dataset)
- [Architecture & Deep Learning Pipeline](#architecture--deep-learning-pipeline)
  - [Preprocessing & Tokenization](#preprocessing--tokenization)
  - [Addressing Class Imbalance](#addressing-class-imbalance)
  - [Model Progression & Comparison](#model-progression--comparison)
  - [BiGRU Final Architecture](#bigru-final-architecture)
- [Web Application & API](#web-application--api)
  - [API Endpoints](#api-endpoints)
  - [Frontend Interface](#frontend-interface)
- [Project Directory Structure](#project-directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Evaluation & Results](#evaluation--results)
- [Future Improvements](#future-improvements)

---

## Overview & Key Features

- **Multi-Class Emotion Classification**: Detects 6 nuanced psychological states from arbitrary text input.
- **Deep Sequence Modeling**: Evaluated multiple architectures (**Simple RNN**, **LSTM**, **GRU**, and **Bidirectional GRU**), comparing hidden state flow and bidirectional representations.
- **High Accuracy**: Reached **92.25% accuracy** with an evaluation loss of **0.2099** on the test split.
- **Production-Ready FastAPI Server**: Asynchronous startup life-cycle, model caching in memory, strict input/output validation with Pydantic schemas, and structured error handling.
- **Interactive Modern Web UI**: Real-time sentiment/emotion breakdown, confidence score bars for all 6 classes, example triggers, health status ping, and responsive dark-mode styling.

---

## Emotions Classified

| Emotion | Emoji | Description |
| :--- | :---: | :--- |
| **Joy** | 😄 | Feelings of happiness, excitement, fulfillment, and contentment |
| **Sadness** | 😢 | Feelings of sorrow, loss, grief, and disappointment |
| **Love** | ❤️ | Expressions of affection, appreciation, empathy, and romantic warmth |
| **Anger** | 😠 | Frustration, annoyance, resentment, or hostility |
| **Fear** | 😨 | Anxiety, panic, vulnerability, or apprehension |
| **Surprise** | 😲 | Shock, astonishment, unexpected events, and awe |

---

## Dataset

The models are trained and validated on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset from Hugging Face:
- **Corpus**: Millions of self-reported emotional experiences extracted from social text posts.
- **Splits**: Standard train, validation, and test splits.
- **Vocabulary**: Top 10,000 frequent terms indexed with sequence padding applied up to `maxlen=50`.

---

## Architecture & Deep Learning Pipeline

```
Raw Input Text
      │
      ▼
Text Preprocessing (lowercase, regex strip punctuation & extra whitespaces)
      │
      ▼
Tokenizer (10,000 vocabulary index) -> Sequence of Integers
      │
      ▼
Post-Padding (`maxlen = 50`)
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│              Bidirectional GRU Architecture             │
├─────────────────────────────────────────────────────────┤
│ 1. Embedding Layer (dim: 300, input_length: 50)         │
│ 2. Bidirectional GRU (128 units, return_sequences=True) │
│ 3. Dropout (0.50)                                       │
│ 4. Bidirectional GRU (64 units)                         │
│ 5. Dropout (0.50)                                       │
│ 6. Dense Layer (6 units, Softmax Activation)            │
└─────────────────────────────────────────────────────────┘
      │
      ▼
Softmax Probability Distribution -> Predicted Emotion & Confidence
```

### Preprocessing & Tokenization
1. **Normalization**: Lowercased text, contraction/apostrophe removal, non-alphanumeric character cleanup, and whitespace collapsing.
2. **Tokenization**: Trained Keras `Tokenizer` indexed on the training split, serialized to [Artifacts/tokenizer.pkl](file:///Users/abhijeetm/Desktop/projects/dl/sentimental_analysis/Artifacts/tokenizer.pkl).
3. **Padded Sequences**: Inputs uniformized using `pad_sequences(maxlen=50, padding='post', truncating='post')`.

### Addressing Class Imbalance
The emotion dataset displays considerable class imbalance (dominated by `joy` and `sadness` vs. minority labels like `surprise` and `love`). To ensure the network doesn't collapse to majority-class predictions:
- Computed inverse class frequencies:
  $$\text{weight}_j = \frac{N}{K \cdot n_j}$$
- Injected `class_weight` into `model.fit()` during training to penalize misclassifications on underrepresented emotion categories.

### Model Progression & Comparison

Through experimental iterations in [sentimental_analysis.ipynb](file:///Users/abhijeetm/Desktop/projects/dl/sentimental_analysis/sentimental_analysis.ipynb), several baseline and sequence networks were benchmarked on identical test sets:

| Model Architecture | Embedding Dim | Test Loss | Test Accuracy | Observations |
| :--- | :---: | :---: | :---: | :--- |
| **Simple RNN** | 128 | 1.7784 | 18.15% | Suffered from vanishing gradients on 50-step sequences |
| **Standard GRU** | 128 | 1.7846 | 11.65% | Underperformed with unidirectional flow under default regularizers |
| **Standard LSTM** | 128 | 0.4079 | 88.00% | Successfully retained long-term sequence dependencies |
| **Bidirectional GRU** (Selected) | **300** | **0.2099** | **92.25%** | Forward + backward context with higher embedding capacity |

### BiGRU Final Architecture

```python
BiGRU = Sequential([
    Embedding(input_dim=10000, output_dim=300, input_length=50),
    Bidirectional(GRU(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(GRU(64)),
    Dropout(0.5),
    Dense(6, activation='softmax')
])
```

- **Optimizer**: Adam
- **Loss Function**: `sparse_categorical_crossentropy`
- **Regularization**: 50% spatial dropout between recurrent stages + EarlyStopping monitoring validation loss.

---

## Web Application & API

The trained model is served using a production-grade **FastAPI** service with a client-side frontend.

### API Endpoints

#### 1. Server Health Check
- **`GET /health`**
- **Response**:
  ```json
  {
    "status": "Server is running",
    "model_loaded": true
  }
  ```

#### 2. Emotion Inference
- **`POST /predict`**
- **Payload**:
  ```json
  {
    "text": "I can't believe we actually won first place after months of preparation!"
  }
  ```
- **Response**:
  ```json
  {
    "text": "I can't believe we actually won first place after months of preparation!",
    "predicted_emotion": "joy",
    "confidence": 0.9842,
    "all_probabilites": {
      "sadness": 0.0012,
      "joy": 0.9842,
      "love": 0.0063,
      "anger": 0.0021,
      "fear": 0.0034,
      "surprise": 0.0028
    }
  }
  ```

### Frontend Interface
- **Interactive Classifier**: Real-time input with live character counter (up to 2000 characters).
- **Preset Prompts**: Pre-configured sample sentences for each emotion.
- **Visual Confidence Gauges**: Animated progress bars showing the probability distribution across all 6 emotions.
- **Responsive Layout**: Designed with clean neutral zinc dark tones, typography from Google Fonts (Inter & JetBrains Mono), and accessible UI states.

---

## Project Directory Structure

```plaintext
sentimental_analysis/
├── Artifacts/
│   ├── BiGRU_Model.keras         # Serialized Keras trained BiGRU model
│   └── tokenizer.pkl             # Serialized fitted Keras Tokenizer
├── static/
│   ├── index.html                # Frontend UI interface
│   ├── style.css                 # Custom styling & dark theme
│   └── script.js                 # UI logic, API calls, and animations
├── sentimental_analysis.ipynb    # Training notebook (EDA, training, comparisons)
├── main.py                       # FastAPI application & inference server
├── .gitignore                    # Ignored artifacts & cache
└── README.md                     # Documentation
```

---

## Getting Started

### Prerequisites
- Python 3.9, 3.10, or 3.11
- `pip` or conda package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhijeetm-07/sentimental_analysis.git
   cd sentimental_analysis
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install fastapi uvicorn tensorflow keras numpy pydantic
   ```

### Running the Application

1. **Launch the FastAPI server:**
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

2. **Access the application:**
   - **Interactive Web UI**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
   - **Interactive Swagger API Docs**: Explore [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
   - **ReDoc API Documentation**: Explore [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).

---

## Evaluation & Results

- **BiGRU Final Test Accuracy**: `92.25%`
- **BiGRU Final Test Loss**: `0.2099`
- Bidirectional context propagation resolved lexical ambiguities (e.g., distinguishing between fear and surprise in abrupt sentences, or recognizing conditional happiness).
- Regularization with 50% dropout and early stopping prevented overfitting on dominant classes.

---

## Future Improvements

- [ ] Fine-tuning Transformer-based architectures (DistilBERT / RoBERTa) for multi-label emotion recognition.
- [ ] Exporting the Keras model to ONNX runtime for ultra-low latency CPU inference.
- [ ] Adding Docker containerization and Docker Compose setup for one-click deployment.
- [ ] Multi-lingual support using multilingual sentence embeddings.
