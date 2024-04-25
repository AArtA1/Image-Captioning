import torch
from torch import nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism allows the model to focus on different parts of the input sequence,
    performing self-attention multiple times in parallel. This class implements the multi-headed attention
    mechanism as described in the "Attention is All You Need" paper.

    Parameters:
        heads (int): Number of attention heads.
        d_model (int): Dimensionality of the input vectors.
        dropout (float, optional): Dropout probability to apply within the attention mechanism. Default is 0.1.

    Inputs:
        query (torch.Tensor): Input tensor representing queries with shape (batch_size, max_len, d_model).
        key (torch.Tensor): Input tensor representing keys with shape (batch_size, max_len, d_model).
        value (torch.Tensor): Input tensor representing values with shape (batch_size, max_len, d_model).
        mask (torch.Tensor): Mask tensor to indicate which elements should be ignored with shape (batch_size, 1, 1, max_words).

    Outputs:
        torch.Tensor: Tensor resulting from the multi-headed attention mechanism with shape (batch_size, max_len, d_model).
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # (batch_size, seq_len, d_model)
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # Split the heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Calculate scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads and linearly transform
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  # (batch_size, seq_len, d_model)
        output = self.out_linear(context)
        return output  


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, feedforward_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head self-attention
        attention_output = self.self_attention(x, x, x, mask=mask)
        # Add and norm
        x = self.layer_norm1(x + self.dropout(attention_output))
        # Position-wise feedforward network
        feedforward_output = self.feedforward(x)
        # Add and norm
        x = self.layer_norm2(x + self.dropout(feedforward_output))
        return x


class  DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, feedforward_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_mask=None, encoder_mask=None):
        # Self-attention
        self_attention_output = self.self_attention(x, x, x, mask=self_mask)
        x = self.layer_norm1(x + self.dropout(self_attention_output))
        
        # Encoder-decoder attention
        encoder_attention_output = self.encoder_attention(x, encoder_output, encoder_output, mask=encoder_mask)
        x = self.layer_norm2(x + self.dropout(encoder_attention_output))
        
        # Position-wise feedforward network
        feedforward_output = self.feedforward(x)
        x = self.layer_norm3(x + self.dropout(feedforward_output))
        
        return x
