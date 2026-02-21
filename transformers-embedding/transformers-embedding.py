import torch
import torch.nn as nn
import math
import numpy as np

def tanh(x):
    x = np.asarray(x, dtype=float)
    return np.tanh(x)

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    return nn.Embedding(vocab_size, d_model)

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    return embedding(tokens) * math.sqrt(d_model)