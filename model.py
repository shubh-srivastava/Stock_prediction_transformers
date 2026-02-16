import torch
import torch.nn as nn


class MultiStockTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 num_stocks,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1):

        super().__init__()

        self.feature_embedding = nn.Linear(input_dim, d_model)
        self.stock_embedding = nn.Embedding(num_stocks, d_model)

        self.positional_encoding = nn.Parameter(torch.randn(1, 30, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, stock_id):

        x = self.feature_embedding(x)

        stock_embed = self.stock_embedding(stock_id)
        stock_embed = stock_embed.unsqueeze(1)

        x = x + stock_embed
        x = x + self.positional_encoding[:, :x.size(1), :]

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.fc(x)
        x = self.tanh(x)
        return x.squeeze()

