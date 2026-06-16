# BERT Question Answering App

A Question Answering (QA) web application built with Streamlit and a fine-tuned BERT model on a modified SQuAD 2.0 dataset.

Users can provide a context paragraph and ask a question. The model extracts the most relevant answer from the context or predicts that no answer exists.

---

## Demo Streamlit App

https://fast-q-a-squad.streamlit.app/

---

## Features

- Extractive Question Answering
- Support for unanswerable questions
- Confidence score display

---

## Dataset

### Hugging Face Dataset

https://huggingface.co/datasets/Vu74269/SQuAD2.0_balanced

Description: 

- An extended version of SQuAD 2.0 created by adding 50K+ unanswerable questions (from other contexts within the same article). The sole purpose is to improve model's ability to detect unanswerable questions.

Original dataset:

- https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/

## Model

### Hugging Face Model

https://huggingface.co/Vu74269/bert-squadv2-base

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
git clone https://github.com/Vu74269/Fast-QA.git
cd Fast-QA
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
The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.
```

### Question

```text
When were the Normans in Normandy?
```

### Answer

```text
10th and 11th centuries
```

---

## Evaluation

Validation Dataset:

- SQuAD 2.0 Validation Set

Metrics:

| Metric | Score |
|----------|----------|
| Exact Match (EM) | 74.404 |
| F1 Score | 77.633 |

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
