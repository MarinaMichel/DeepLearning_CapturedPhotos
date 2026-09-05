"""
Seq2Seq (BiLSTM encoder + attention decoder) that translates a sequence of
per-frame MediaPipe keypoints into a gloss (word) sequence.

Same architecture as your notebook, moved here so train.py and the FastAPI
backend can both import it instead of duplicating the class definitions.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_dim=1662, hid_dim=512, n_layers=2, dropout=0.3):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.input_proj = nn.Linear(input_dim, hid_dim)
        self.lstm = nn.LSTM(hid_dim, hid_dim, n_layers,
                             batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.fc_h = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, x, lengths):
        x = self.dropout(self.input_proj(x))

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        hidden = hidden.view(self.n_layers, 2, hidden.size(1), hidden.size(2))
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)
        hidden = self.fc_h(hidden)

        cell = cell.view(self.n_layers, 2, cell.size(1), cell.size(2))
        cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)
        cell = self.fc_c(cell)

        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim=512, dec_hid_dim=512):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size, src_len, _ = encoder_outputs.shape

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        weights = torch.softmax(attention, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, weights


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim=256, enc_hid_dim=512,
                 dec_hid_dim=512, n_layers=2, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(enc_hid_dim, dec_hid_dim)

        self.lstm = nn.LSTM(emb_dim + (enc_hid_dim * 2), dec_hid_dim,
                             n_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(dec_hid_dim + (enc_hid_dim * 2) + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        context, attn_w = self.attention(hidden[-1], encoder_outputs)
        context = context.unsqueeze(1)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))

        return prediction, hidden, cell, attn_w


class SignLanguageTranslator(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        import random
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs


def translate(model, features, vocab, device, max_len=50):
    """Run inference: features (T, 1662) -> predicted gloss string."""
    model.eval()

    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)

    features = features.unsqueeze(0).to(device)
    src_len = torch.LongTensor([features.shape[1]]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(features, src_len)

    inputs = torch.LongTensor([vocab.word2idx["<sos>"]]).to(device)
    outputs = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell, _ = model.decoder(inputs, hidden, cell, encoder_outputs)

        pred = output.argmax(1).item()

        if pred == vocab.word2idx["<eos>"]:
            break

        outputs.append(vocab.idx2word[pred])
        inputs = torch.LongTensor([pred]).to(device)

    return " ".join(outputs)
