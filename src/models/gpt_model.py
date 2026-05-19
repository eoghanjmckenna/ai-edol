"""
GPT model architecture for smart meter data generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import logging


class CustomTransformerDecoderLayer(nn.Module):
    """Custom Transformer Decoder Layer with prenorm architecture."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        """
        Initialize CustomTransformerDecoderLayer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
        """
        super(CustomTransformerDecoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, tgt: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the layer.
        
        Args:
            tgt: Target tensor
            tgt_mask: Attention mask
            
        Returns:
            Transformed tensor
        """
        # Prenorm: apply layer normalisation before self-attention
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # Prenorm: apply layer normalisation before feedforward network
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        
        return tgt


class GPTSmartMeterModel(nn.Module):
    """
    GPT model for smart meter data generation with single-output mode.

    This model can be configured to generate either electricity OR gas,
    enabling separate specialized models for each target variable while
    maintaining cross-fuel context through input features.

    Architecture Modes:
    - Electricity Model: Generates electricity conditioned on gas history + context
    - Gas Model: Generates gas conditioned on electricity history + context

    Feature Ordering (11 features):
    - Index 0: Electricity (INPUT AND OUTPUT for elec model, INPUT for gas model)
    - Index 1: Gas (INPUT for elec model, INPUT AND OUTPUT for gas model)
    - Index 2: Temperature (INPUT for both)
    - Index 3: Solar radiation (INPUT for both)
    - Index 4: Half-hour of day (INPUT for both, 1-48)
    - Index 5: Day of week (INPUT for both, 0-6)
    - Index 6: Month (INPUT for both, 1-12)
    - Index 7: building_type (INPUT CONDITIONING for both)
    - Index 8: age_built (INPUT CONDITIONING for both)
    - Index 9: num_rooms (INPUT CONDITIONING for both)
    - Index 10: num_occs (INPUT CONDITIONING for both)

    Simplified Conditioning Approach:
    - Reduced from 14 to 4 conditioning variables for improved fidelity
    - Selected variables: building_type, age_built, num_rooms, num_occs
    - No masking - all conditioning variables always present

    Usage:
    - Two separate models trained independently (electricity and gas)
    - Both models see cross-fuel context during training (teacher forcing)
    - During inference, DualModelGenerator coordinates parallel generation
    - Preserves elec-gas correlations through historical context
    """
    
    def __init__(
        self,
        vocab_sizes: List[int],
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        context_length: int,
        dropout: float = 0.1,
        output_variable: str = 'electricity'
    ):
        """
        Initialize GPTSmartMeterModel.

        Args:
            vocab_sizes: List of vocabulary sizes for each input feature (11 features).
                        Computed from tokenizer metadata, not configured manually.
            d_model: Model dimension (embedding and hidden layer size)
            nhead: Number of attention heads
            num_layers: Number of transformer decoder layers
            dim_feedforward: Feedforward network dimension
            context_length: Maximum sequence length (e.g., 2048)
            dropout: Dropout probability for regularization
            output_variable: Target variable to generate - either 'electricity' or 'gas'.
                           Determines which output head is created.

        Raises:
            ValueError: If output_variable is not 'electricity' or 'gas'
            AssertionError: If vocab_sizes does not have exactly 11 elements
        """
        super(GPTSmartMeterModel, self).__init__()

        # Validate vocab_sizes length (simplified conditioning: 4 variables instead of 14)
        assert len(vocab_sizes) == 11, f"Expected 11 vocab_sizes, got {len(vocab_sizes)}"
        
        self.output_variable = output_variable

        self.num_input_features = len(vocab_sizes)
        self.d_model = d_model
        self.context_length = context_length
        
        # Embedding layers for each input feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for vocab_size in vocab_sizes
        ])
        
        # Learnable positional embedding
        self.pos_embedding = nn.Embedding(context_length, d_model)
        
        # Register lower triangular mask as buffer for efficiency
        self.register_buffer(
            'tril_mask', 
            torch.tril(torch.ones(context_length, context_length))
        )
        
        # Custom Transformer Decoder layers
        self.transformer_decoder_layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) 
            for _ in range(num_layers)
        ])
        
        if output_variable == 'electricity':
            self.output_layer = nn.Linear(d_model, vocab_sizes[0])
        elif output_variable == 'gas':
            self.output_layer = nn.Linear(d_model, vocab_sizes[1])
        else:
            raise ValueError(f"Unsupported output variable: {output_variable}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, num_input_features]
               where num_input_features = 11 (elec, gas, temp, solar, hh, dow, month, + 4 conditioning)

        Returns:
            Output logits of shape [batch_size, seq_len, vocab_size]
            where vocab_size is either vocab_sizes[0] (electricity) or vocab_sizes[1] (gas)
            depending on output_variable configuration.

        Note:
            This method only performs forward pass. Autoregressive generation is handled
            by DualModelGenerator which coordinates both electricity and gas models.
        """
        # Embed each feature separately and sum them
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            max_index = x[:, :, i].max().item()
            vocab_size = emb_layer.num_embeddings
            if max_index >= vocab_size:
                logging.debug(f'Feature {i} has index out of range. Max: {max_index}, vocab: {vocab_size}')
                raise ValueError(f'Input feature {i} has index ({max_index}) exceeding vocab size ({vocab_size})')
            embeddings.append(emb_layer(x[:, :, i].long()))
        
        x = sum(embeddings)  # [batch_size, context_length, d_model]
        
        # Add learnable positional embedding
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_encoding = self.pos_embedding(positions)
        x = x + pos_encoding
        
        # Generate causal mask
        subsequent_mask = self.tril_mask[:seq_len, :seq_len]
        subsequent_mask = subsequent_mask.masked_fill(subsequent_mask == 0, float('-inf'))
        
        # Pass through transformer layers
        for layer in self.transformer_decoder_layers:
            x = layer(x, tgt_mask=subsequent_mask)
            
        # Project to output vocabularies (electricity or gas depending on config)
        output = self.output_layer(x)          
        
        return output
    