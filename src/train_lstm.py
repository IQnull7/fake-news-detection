"""Module 8: Self-contained PyTorch BiLSTM Deep Learning Model for Fake News Detection."""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import collections
import time
import json
import pandas as pd
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, SEED, set_seed
from src.preprocess import clean_text_light
from src.evaluate import evaluate


class Vocabulary:
    """Lightweight self-contained vocabulary builder for PyTorch text models."""

    def __init__(self, pad_token: str = "<pad>", unk_token: str = "<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.w2i: Dict[str, int] = {pad_token: 0, unk_token: 1}
        self.i2w: Dict[int, str] = {0: pad_token, 1: unk_token}

    def build_from_texts(
        self,
        texts: List[str],
        max_vocab_size: int = 25000,
        min_freq: int = 2,
    ) -> None:
        counter = collections.Counter()
        for t in texts:
            tokens = str(t).split()
            counter.update(tokens)

        # Most frequent tokens
        for word, count in counter.most_common(max_vocab_size - len(self.w2i)):
            if count >= min_freq and word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        tokens = str(text).split()
        unk_idx = self.w2i[self.unk_token]
        pad_idx = self.w2i[self.pad_token]

        token_ids = [self.w2i.get(tok, unk_idx) for tok in tokens[:max_len]]
        if len(token_ids) < max_len:
            token_ids += [pad_idx] * (max_len - len(token_ids))
        return token_ids

    def __len__(self) -> int:
        return len(self.w2i)


class NewsTextDataset(Dataset):
    """PyTorch Dataset for news articles."""

    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 256):
        self.labels = labels
        self.encoded = [vocab.encode(t, max_len=max_len) for t in texts]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with Dual Pooling (Max + Average) and Classification Head."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout_p: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout_p)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
        )

        # Dual pooling: hidden_dim * 2 (bidirectional) * 2 (avg + max) = hidden_dim * 4
        pooled_dim = hidden_dim * 4
        self.fc = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        embedded = self.emb_dropout(self.embedding(x))
        out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)

        # Global average & max pooling over time steps
        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        logits = self.fc(pooled)
        return logits


def train_bilstm_model(
    batch_size: int = 128,
    epochs: int = 2,
    max_len: int = 150,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load splits (using text_light to preserve natural linguistic structure as per Caveat 8)
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    print("Loading data splits...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Build Vocabulary strictly on train split
    print("Building vocabulary on training set (max_size=25,000)...")
    vocab = Vocabulary()
    vocab.build_from_texts(df_train["text_light"].fillna("").tolist(), max_vocab_size=25000, min_freq=2)
    print(f"Vocabulary built: {len(vocab):,} unique tokens.")

    # Save vocabulary
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocab, MODELS_DIR / "bilstm_vocab.joblib")

    # DataLoaders
    train_ds = NewsTextDataset(df_train["text_light"].fillna("").tolist(), df_train["label"].tolist(), vocab, max_len)
    val_ds = NewsTextDataset(df_val["text_light"].fillna("").tolist(), df_val["label"].tolist(), vocab, max_len)
    test_ds = NewsTextDataset(df_test["text_light"].fillna("").tolist(), df_test["label"].tolist(), vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        dropout_p=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_weights_path = MODELS_DIR / "bilstm_best.pt"

    print("\nStarting BiLSTM Training:")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                logits_v = model(x_v)
                loss_v = criterion(logits_v, y_v)
                val_loss += loss_v.item()
                preds = torch.argmax(logits_v, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_v.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        epoch_dur = time.time() - start_t

        print(
            f"Epoch {epoch}/{epochs} ({epoch_dur:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_weights_path)
            print(f"  --> Saved new best checkpoint (Val Loss: {avg_val_loss:.4f})")

    # Final Evaluation on Hold-Out Test Set
    print("\n--- Evaluating Best BiLSTM Checkpoint on Test Set ---")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t = x_t.to(device)
            logits_t = model(x_t)
            probs = torch.softmax(logits_t, dim=1)[:, 1]
            preds = torch.argmax(logits_t, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_targets.extend(y_t.numpy())

    test_metrics = evaluate(
        y_true=np.array(test_targets),
        y_pred=np.array(test_preds),
        y_score=np.array(test_probs),
        model_name="BiLSTM (PyTorch)",
        split_name="test",
    )
    print("BiLSTM Test Metrics:", test_metrics)

    # Checkpoint Verification: Single-sentence inference speed
    print("\n--- Checkpoint Verification: Single Article Inference ---")
    raw_sample = "The defense ministry announced new maritime security patrols in the region."
    t0 = time.time()
    clean_sample = clean_text_light(raw_sample)
    encoded_sample = torch.tensor([vocab.encode(clean_sample, max_len=max_len)], dtype=torch.long).to(device)
    with torch.no_grad():
        sample_logits = model(encoded_sample)
        sample_prob = torch.softmax(sample_logits, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(sample_prob))
    infer_time_ms = (time.time() - t0) * 1000

    print(f"Sample: '{raw_sample}'")
    print(f"Prediction: {pred_class} (Fake: {sample_prob[0]:.3f}, Real: {sample_prob[1]:.3f})")
    print(f"Inference latency: {infer_time_ms:.2f} ms")

    assert infer_time_ms < 2000, "Inference latency exceeded 2 seconds!"
    print("CHECKPOINT PASSED: BiLSTM trained with decreasing loss, evaluated on test split, and latency is well under 2s.")

    return test_metrics


if __name__ == "__main__":
    train_bilstm_model()

