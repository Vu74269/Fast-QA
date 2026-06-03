# 🤖 BERT Question Answering App

A Question Answering (QA) web application built with Streamlit and a fine-tuned BERT model on the SQuAD 2.0 dataset.

Users can provide a context paragraph and ask a question. The model extracts the most relevant answer from the context or predicts that no answer exists.

---

## Hugging Face Model

https://huggingface.co/Vu74269/bert-squadv2-base

---

## Features

- Extractive Question Answering
- Support for unanswerable questions
- Confidence score display

---

## Model

Base Model:

- BERT Base Uncased

Dataset:

- SQuAD 2.0

Task:

- Extractive Question Answering

---

## Installation

Clone repository:

```bash
git clone https://github.com/Vu74269/Fast-QandA.git
cd Fast-QandA
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Example

### Context

```text
The Normans were the people who gave their name to Normandy, a region in France.
```

### Question

```text
What region in France was named after the Normans?
```

### Answer

```text
Normandy
```

---

## Evaluation

Validation Dataset:

- SQuAD 2.0 Validation Set

Metrics:

| Metric | Score |
|----------|----------|
| Exact Match (EM) | 64.36 |
| F1 Score | 71.37 |

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
