from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch
import streamlit as st

MODEL_NAME = "Vu74269/bert-squadv2-extend-base"

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForQuestionAnswering.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    return tokenizer, model