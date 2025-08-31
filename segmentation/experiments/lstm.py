import torch
import torch.nn as nn

class ModernEmbeddingLSTM(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
        use_attention=True,
    ):
        super().__init__()

        # Calculate directions
        self.directions = 2 if bidirectional else 1
        self.use_attention = use_attention

        # LSTM layer with dropout between layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * self.directions)
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * self.directions, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        # Output projection to match embedding dimension
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * self.directions, hidden_dim * 2),
            nn.GELU(),  # Modern activation function
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, embedding_dim),
        )
        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    nn.init.orthogonal_(param)  # Orthogonal initialization for RNNs
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(
                        param
                    )  # Xavier for linear layers (2D+ weights)
                else:
                    nn.init.normal_(
                        param, mean=0.0, std=0.02
                    )  # Normal init for 1D weights
            elif "bias" in name:
                nn.init.zeros_(param)  # Initialize biases to zero

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            return_attention: Whether to return attention weights

        Returns:
            Predicted next embedding of shape [batch_size, embedding_dim]
            Optionally, attention weights if return_attention=True
        """
        # x shape: [B, T, embedding_dim]
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Run through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: [B, T, hidden_dim * directions]

        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)

        if self.use_attention:
            # Calculate attention scores
            attention_scores = self.attention(lstm_out).squeeze(-1)  # [B, T]
            attention_weights = torch.softmax(attention_scores, dim=1)  # [B, T]

            # Apply attention to get context vector
            attention_weights = attention_weights.unsqueeze(-1)  # [B, T, 1]
            context = torch.sum(
                attention_weights * lstm_out, dim=1
            )  # [B, hidden_dim * directions]
        else:
            # Just use the last output
            context = lstm_out[:, -1]  # [B, hidden_dim * directions]

        # Project to embedding dimension
        next_emb = self.proj(context)  # [B, embedding_dim]

        if return_attention and self.use_attention:
            return next_emb, attention_weights.squeeze(-1)
        return next_emb

    def predict_sequence(self, initial_sequence, steps=5):
        """
        Generate a sequence of future embeddings

        Args:
            initial_sequence: Initial sequence of embeddings [B, T, embedding_dim]
            steps: Number of future steps to predict

        Returns:
            Sequence of predicted embeddings [B, steps, embedding_dim]
        """
        device = next(self.parameters()).device
        batch_size = initial_sequence.shape[0]
        emb_dim = initial_sequence.shape[2]

        # Start with the initial sequence
        current_seq = initial_sequence
        predictions = torch.zeros(batch_size, steps, emb_dim, device=device)

        # Autoregressive prediction
        for i in range(steps):
            # Predict next embedding
            next_emb = self(current_seq)
            predictions[:, i] = next_emb

            # Update sequence for next prediction (drop oldest, add newest)
            if i < steps - 1:  # No need to update for the last step
                current_seq = torch.cat(
                    [current_seq[:, 1:], next_emb.unsqueeze(1)], dim=1
                )

        return predictions