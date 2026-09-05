"""
Step 3: Train the seq2seq sign-language translator on the quarter-subset
of WLASL, using the features extracted by extract_all_features.py.

Usage:
    python train.py
    python train.py --csv wlasl_quarter.csv --features_dir extracted_features

Saves:
    models/best_sign_language_model.pt   -- best checkpoint by val loss
    models/vocab.pkl                     -- vocabulary (needed for inference)
"""
import argparse
import os
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import Encoder, Decoder, SignLanguageTranslator, translate
from vocab import Vocabulary

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class SignLanguageDataset(Dataset):
    def __init__(self, df, features_dir, vocab):
        self.df = df.reset_index(drop=True)
        self.features_dir = features_dir
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        text = str(row["gloss"])

        feat_path = os.path.join(self.features_dir, f"{video_id}.npy")
        features = np.load(feat_path).astype(np.float32)

        tokens = [self.vocab.word2idx["<sos>"]]
        tokens += self.vocab.numericalize(text)
        tokens.append(self.vocab.word2idx["<eos>"])

        return torch.FloatTensor(features), torch.LongTensor(tokens)


def collate_fn(batch):
    features, targets = zip(*batch)

    feat_lengths = [f.shape[0] for f in features]
    max_feat_len = max(feat_lengths)
    feat_dim = features[0].shape[1]
    padded_feats = torch.zeros(len(features), max_feat_len, feat_dim)
    for i, f in enumerate(features):
        padded_feats[i, :f.shape[0]] = f

    tgt_lengths = [t.shape[0] for t in targets]
    max_tgt_len = max(tgt_lengths)
    padded_tgts = torch.full((len(targets), max_tgt_len), fill_value=0)
    for i, t in enumerate(targets):
        padded_tgts[i, :t.shape[0]] = t

    return (padded_feats, padded_tgts,
            torch.LongTensor(feat_lengths), torch.LongTensor(tgt_lengths))


def train_epoch(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        src, trg, src_len, trg_len = [x.to(device) for x in batch]

        optimizer.zero_grad()
        output = model(src, src_len, trg, teacher_forcing_ratio=0.5)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            src, trg, src_len, trg_len = [x.to(device) for x in batch]
            output = model(src, src_len, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="wlasl_quarter.csv")
    parser.add_argument("--features_dir", default="extracted_features")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--dec_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.csv)
    df["has_features"] = df["video_id"].apply(
        lambda vid: os.path.exists(os.path.join(args.features_dir, f"{vid}.npy"))
    )
    df_ready = df[df["has_features"]].copy().drop(columns=["has_features"])
    print(f"Total videos in CSV: {len(df)} | with extracted features: {len(df_ready)}")

    if len(df_ready) == 0:
        raise ValueError(
            f"No .npy files found in '{args.features_dir}'. "
            "Run extract_all_features.py first!"
        )

    all_sentences = df_ready["gloss"].astype(str).tolist()
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(all_sentences)
    print(f"Vocabulary size: {len(vocab)}")

    with open(os.path.join(args.model_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    train_df = df_ready.sample(frac=0.9, random_state=SEED)
    val_df = df_ready.drop(train_df.index)

    train_loader = DataLoader(
        SignLanguageDataset(train_df, args.features_dir, vocab),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        SignLanguageDataset(val_df, args.features_dir, vocab),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    enc = Encoder(input_dim=1662, hid_dim=args.hid_dim,
                  n_layers=args.enc_layers, dropout=args.dropout)
    dec = Decoder(output_dim=len(vocab), emb_dim=256, enc_hid_dim=args.hid_dim,
                  dec_hid_dim=args.hid_dim, n_layers=args.dec_layers, dropout=args.dropout)
    model = SignLanguageTranslator(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.model_dir, "best_sign_language_model.pt")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.clip, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print("  -> Saved new best model")

        print(f"Epoch {epoch+1:02d}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

    print("\n--- Sample Translation ---")
    sample_video_id = val_df.iloc[0]["video_id"]
    sample_text = val_df.iloc[0]["gloss"]
    sample_features = np.load(os.path.join(args.features_dir, f"{sample_video_id}.npy"))

    model.load_state_dict(torch.load(ckpt_path))
    translation = translate(model, sample_features, vocab, device)

    print(f"Video ID : {sample_video_id}")
    print(f"Target   : {sample_text}")
    print(f"Predicted: {translation}")


if __name__ == "__main__":
    main()
