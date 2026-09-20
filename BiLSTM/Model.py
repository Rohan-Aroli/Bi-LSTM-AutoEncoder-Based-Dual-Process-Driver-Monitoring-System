import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        latent = torch.cat((forward_hidden, backward_hidden), dim=1)
        return latent

class BiLSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, latent, seq_len):
        x = latent.unsqueeze(1).repeat(1, seq_len, 1)
        lstm_out, _ = self.lstm(x)
        reconstructed = self.fc(lstm_out)
        return reconstructed

class BiLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = BiLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        latent_dim = hidden_dim * 2
        self.decoder = BiLSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x):
        seq_len = x.size(1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent, seq_len)
        return reconstructed