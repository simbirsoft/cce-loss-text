"""
Runs experiments and generates report
"""


import argparse
import numpy as np
import os
import pickle
from sklearn.metrics import balanced_accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from torch import autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
from cce_loss import complement_cross_entropy_loss


def read_data(dataset_folder, ratio_0, ratio_p1, batch_size, device):
    phrase_field = Field(sequential=True, use_vocab=True, tokenize="spacy", lower=True, 
        include_lengths=True, batch_first=True)
    sentiment_field = Field(sequential=False, use_vocab=False, batch_first=True)
    fields = [
        ("Phrase", phrase_field),
        ("Sentiment", sentiment_field)
    ]

    train_filename = f"train_{ratio_0:.2f}_{ratio_p1:.2f}.csv"
    val_filename = f"val_{ratio_0:.2f}_{ratio_p1:.2f}.csv"
    test_filename = f"test_{ratio_0:.2f}_{ratio_p1:.2f}.csv"
    train_ds, val_ds, test_ds = TabularDataset.splits(dataset_folder, 
        format="CSV", fields=fields, skip_header=True,
        train=train_filename, validation=val_filename, test=test_filename)

    train_it = BucketIterator(train_ds, batch_size,
        sort_key=lambda x: len(x.Phrase), device=device, 
        sort=True, sort_within_batch=True)
    val_it = BucketIterator(val_ds, batch_size,
        sort_key=lambda x: len(x.Phrase), device=device, 
        sort=True, sort_within_batch=True)
    test_it = BucketIterator(test_ds, batch_size,
        sort_key=lambda x: len(x.Phrase), device=device, 
        sort=True, sort_within_batch=True)

    phrase_field.build_vocab(train_ds)

    return len(phrase_field.vocab), train_it, val_it, test_it

def y_to_ohe(y, n_classes):
    return F.one_hot(y, n_classes)

def one_hot_cross_entropy_loss(y_hat, y):
    return F.cross_entropy(y_hat, torch.argmax(y, dim=1))

def train_model(model, n_classes, criterion, optimizer, 
                train_it, val_it, epochs, early_stopping_patience,
                checkpoint_file, tb_writer):
    best_val_loss = float("inf")
    best_val_epoch = 0

    tb_writer.add_text("Monitor", f"Num train batches: {len(train_it)}")
    tb_writer.add_text("Monitor", f"Num val batches: {len(val_it)}")

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for ((x, len_x), y), _ in train_it:
            y = y_to_ohe(y, n_classes)
            y_pred = model(x, len_x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_it)
        tb_writer.add_scalar("Loss/Train", train_loss, epoch)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for ((x, len_x), y), _ in val_it:
                y = y_to_ohe(y, n_classes)
                y_pred = model(x, len_x)
                loss = criterion(y_pred, y)

                val_loss += loss.item()
            val_loss /= len(val_it)
        tb_writer.add_scalar("Loss/Val", val_loss, epoch)
        
        if val_loss < best_val_loss:
            tb_writer.add_text("Monitor", "New best model", epoch)
            torch.save(model.state_dict(), checkpoint_file)
            best_val_loss = val_loss
            best_val_epoch = epoch
        if (epoch - best_val_epoch) > early_stopping_patience:
            tb_writer.add_text("Monitor", "Early stopping", epoch)
            break

    model.load_state_dict(torch.load(checkpoint_file))
    return model

def evaluate_model(model, n_classes, test_it, tb_writer):
    model.eval()
    ys = []
    y_preds = []

    with torch.no_grad():
        for ((x, len_x), y), _ in test_it:
            ys.extend(y.cpu().numpy())
            y = y_to_ohe(y, n_classes)
            y_pred = model(x, len_x)
            y_pred = torch.argmax(y_pred, dim=1)
            y_preds.extend(y_pred.cpu().numpy())

    bacc = balanced_accuracy_score(ys, y_preds, adjusted=True)
    clf_rep = classification_report(ys, y_preds)

    tb_writer.add_text("Score", f"Balanced accuracy: {bacc}")
    tb_writer.add_text("Score", f"Classification report:\n{clf_rep}")

    return bacc, clf_rep

def run_experiment(dataset_folder, ratio_0, ratio_p1, use_cce, 
                   batch_size, lr, epochs, early_stopping_patience,
                   emb_dim, lstm_hidden_dim, emb_dropout_p, lstm_dropout_p, n_classes,
                   checkpoint_file, device):
    tb_writer = SummaryWriter(comment=f"_r0_{ratio_0}_rp1_{ratio_p1}_cce_{use_cce}")
    vocab_size, train_it, val_it, test_it = read_data(dataset_folder, ratio_0, ratio_p1, batch_size, device)
    model = BiLstmClassifier(vocab_size, emb_dim, lstm_hidden_dim, emb_dropout_p, lstm_dropout_p, n_classes).to(device)
    opt = optim.Adam(model.parameters(), lr)
    criterion = complement_cross_entropy_loss if use_cce else one_hot_cross_entropy_loss
    with autograd.detect_anomaly():
        model = train_model(model, n_classes, criterion, opt, train_it, val_it, epochs, early_stopping_patience, checkpoint_file, tb_writer)
    score = evaluate_model(model, n_classes, test_it, tb_writer)
    return score

def dump_scores(ratios, scores, scores_pickle_filename):
    with open(scores_pickle_filename, "wb") as f:
        pickle.dump(scores, f)

class BiLstmClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, emb_dropout_p, lstm_dropout_p, n_classes):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.emb_dropout_p = emb_dropout_p
        self.lstm_dropout_p = lstm_dropout_p
        self.n_classes = n_classes

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim,
            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, n_classes)

    def forward(self, x, len_x):
        batch_len = len(x)

        x = self.emb(x)
        x = F.dropout(x, self.emb_dropout_p)
        
        x = pack_padded_sequence(x, len_x, batch_first=True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x_f = x[:, -1, :self.hidden_dim]
        x_b = x[:, 0, self.hidden_dim:]
        x = torch.cat((x_f, x_b), dim=1)
        x = F.dropout(x, self.lstm_dropout_p)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", help="Folder containing datasets for experiments")
    parser.add_argument("--mid_class_ratio", help="Ratio of a middle-sized class", type=float, default=0.5)
    parser.add_argument("--low_class_ratio", help="Ratio of a low-sized class", type=float, default=0.2)
    parser.add_argument("--use_cce", help="Flag to use Complement cross-entropy loss", action="store_true")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=256)
    parser.add_argument("--lr", help="Learning rate value", type=float, default=3e-4)
    parser.add_argument("--epochs", help="Number of epochs to train for", default=50)
    parser.add_argument("--early_stopping_patience", help="Patience for learning early stopping (epochs)", type=int, default=5)
    parser.add_argument("--gpu", help="Flag to use GPU", action="store_true")
    parser.add_argument("--emb_dim", help="Word embedding dim", type=int, default=300)
    parser.add_argument("--lstm_hidden_dim", help="LSTM's hidden state dim", type=int, default=128)
    parser.add_argument("--emb_dropout_p", help="Dropout rate for embeddings", type=float, default=0.1)
    parser.add_argument("--lstm_dropout_p", help="Dropout rate for LSTM outputs", type=float, default=0.1)
    parser.add_argument("--n_classes", help="Number of classes in the problem", type=int, default=3)
    parser.add_argument("--checkpoint_file", help="File to store checkpoints to", default="./checkpoints/model.ckpt")
    parser.add_argument("--scores_pickle_filename", help="Name of a file to pickle resulting metrics to", default="./scores.pkl")
    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    checkpoint_folder = os.path.dirname(args.checkpoint_file)
    os.makedirs(checkpoint_folder, exist_ok=True)

    ratios = [
        (1, 1),
        (args.mid_class_ratio, args.mid_class_ratio),
        (args.mid_class_ratio, args.low_class_ratio),
        (args.low_class_ratio, args.low_class_ratio),
    ]

    scores = []
    for ratio_0, ratio_p1 in ratios:
        print(f"Running experiments for ratio_0={ratio_0} and ratio_p1={ratio_p1}")
        score = run_experiment(args.dataset_folder, ratio_0, ratio_p1, args.use_cce, 
            args.batch_size, args.lr, args.epochs, args.early_stopping_patience,
            args.emb_dim, args.lstm_hidden_dim, args.emb_dropout_p, args.lstm_dropout_p, args.n_classes,
            args.checkpoint_file, device)
        scores.append(score)
        print(f"Experiment score:")
        print(f"bacc = {score[0]}")
        print(score[1])
    print(f"Done running experiments")
    dump_scores(ratios, scores, args.scores_pickle_filename)
