import torch
import torch.nn as nn
import random

# -----------------------------
# Encoder & Decoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.linears = nn.ModuleList([nn.Linear(C, C) for _ in range(H * W)])
        for lin in self.linears:
            nn.init.eye_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, H*W, C)
        out = torch.stack([lin(x_perm[:, i, :]) for i, lin in enumerate(self.linears)], dim=1)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # back to (B, C, H, W)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, H, W):
        super().__init__()
        self.H, self.W = H, W
        # linear per spatial location
        self.linears = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(H*W)])

    def forward(self, x):
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, H*W, C)
        out = torch.stack([lin(x_perm[:, i, :]) for i, lin in enumerate(self.linears)], dim=1)
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, out_channels, H, W)
        return out

# -----------------------------
# Convolutions
# -----------------------------
class Convolutions(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, kernel_size=3):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(hidden_channels_list)
        assert len(kernel_size) == len(hidden_channels_list)

        self.convs = nn.ModuleList()
        prev_channels = in_channels
        self.kernel_sizes = kernel_size
        for out_channels, k in zip(hidden_channels_list, kernel_size):
            self.convs.append(nn.Conv2d(prev_channels, out_channels, kernel_size=k, padding=0))
            prev_channels = out_channels

    def forward(self, x):
        for conv, k in zip(self.convs, self.kernel_sizes):
            pad_h, pad_w = k // 2, k // 2
            x = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='circular')
            x = conv(x)
            x = torch.relu(x)
        return x

# -----------------------------
# LSTMs
# -----------------------------
class LSTMs(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward_step(self, x_t, hc, skip_fc=False):
        h_list, c_list = hc
        new_h_list, new_c_list = [], []
        input_l = x_t
        for i, cell in enumerate(self.lstm_cells):
            h, c = cell(input_l, (h_list[i], c_list[i]))
            new_h_list.append(h)
            new_c_list.append(c)
            input_l = h

        y = new_h_list[-1] if skip_fc else self.fc(new_h_list[-1])
        return y, (new_h_list, new_c_list)

# -----------------------------
# Zeus2 model
# -----------------------------
class Zeus2(nn.Module):
    def __init__(self, input_shape, cnn_hidden_channels, lstm_layers=2):
        super().__init__()
        C, H, W = input_shape
        self.C, self.H, self.W = C, H, W

        # Encoder & CNN
        self.encoder = Encoder(C, H, W)
        self.cnn = Convolutions(in_channels=C, hidden_channels_list=cnn_hidden_channels)
        self.cnn_out_channels = cnn_hidden_channels[-1]

        # # LSTM
        flattened_size = self.cnn_out_channels * H * W
        # self.lstm = LSTMs(input_dim=flattened_size, hidden_dim=flattened_size, num_layers=lstm_layers)
        
        # Allow smaller hidden_dim
        lstm_hidden_dim = self.cnn_out_channels * H * W // 2000  # reduced hidden_dim, 137,554,329,600 bytes for LSTM cell.
        self.lstm = LSTMs(input_dim=flattened_size, hidden_dim=lstm_hidden_dim, num_layers=lstm_layers)

        # Decoder maps cnn_out_channels -> C
        self.decoder = Decoder(in_channels=self.cnn_out_channels, out_channels=C, H=H, W=W)

    def encode_cnn_flatten(self, x):
        x_enc = self.encoder(x)
        x_cnn = self.cnn(x_enc)
        return x_cnn.flatten(start_dim=1)

    def forward(self, x_seq, m, sampling_prob=1.0):
        n = len(x_seq) - m
        batch_size = x_seq[0].shape[0]
        device = x_seq[0].device

        h = [torch.zeros(batch_size, self.lstm.lstm_cells[0].hidden_size, device=device)
             for _ in range(self.lstm.num_layers)]
        c = [torch.zeros(batch_size, self.lstm.lstm_cells[0].hidden_size, device=device)
             for _ in range(self.lstm.num_layers)]

        preds = []
        input_t = None

        for t in range(len(x_seq) - 1):
            if t < n - 1:
                input_t = x_seq[t]
            else:
                if random.random() < sampling_prob:
                    input_t = x_seq[t]

            x_flat = self.encode_cnn_flatten(input_t)
            skip_fc = t < n - 1
            lstm_out, (h, c) = self.lstm.forward_step(x_flat, (h, c), skip_fc=skip_fc)

            if t >= n - 1:
                # reshape LSTM output to (B, cnn_out_channels, H, W)
                B = batch_size
                pred_cnn = lstm_out.view(B, self.cnn_out_channels, self.H, self.W)
                # map back to original channels
                pred = self.decoder(pred_cnn)
                preds.append(pred)
                input_t = pred

        return preds


# -----------------------------
# Test run
# -----------------------------
if __name__ == "__main__":

    batch_size = 2
    C, H, W = 3, 8, 8
    n, m = 5, 3
    x_seq = [torch.randn(batch_size, C, H, W, requires_grad=True) for _ in range(n+m)]

    model = Zeus2(input_shape=(C, H, W), cnn_hidden_channels=[4, 8], lstm_layers=2)
    preds = model(x_seq, m=m, sampling_prob=0.8)

    print("Number of predictions:", len(preds))
    print("Prediction shape:", preds[0].shape)  # (batch_size, C, H, W)
