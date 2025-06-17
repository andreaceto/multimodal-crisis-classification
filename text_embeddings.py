import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Set device: GPU if available, otherwise CPU
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_bert_embeddings(df, text_column, model_name="bert-base-uncased", batch_size=16, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []

    for i in tqdm(range(0, len(df), batch_size), desc="BERT embeddings"):
        batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**encoded)
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
        embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


def generate_roberta_embeddings(df, text_column, model_name="roberta-base", batch_size=16, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []

    for i in tqdm(range(0, len(df), batch_size), desc="RoBERTa embeddings"):
        batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**encoded)
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


def generate_bertweet_embeddings(df, text_column, model_name="vinai/bertweet-base", batch_size=16, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []

    for i in tqdm(range(0, len(df), batch_size), desc="BERTweet embeddings"):
        batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**encoded)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

    return np.vstack(embeddings)

