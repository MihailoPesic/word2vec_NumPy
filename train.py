"""
Skip-gram word2vec with negative sampling — pure NumPy implementation.

Usage:
    python train.py                          # downloads text8 automatically
    python train.py --data path/to/file.txt  # use your own corpus
    python train.py --epochs 5 --embed_dim 200 --n_neg 10
"""

import argparse
import os
import time
import urllib.request
import zipfile

import numpy as np

from preprocess import load_text, build_vocab, subsample, build_pairs, noise_distribution, build_noise_table
from word2vec import SkipGramNS


TEXT8_URL = 'http://mattmahoney.net/dc/text8.zip'


def download_text8(dest='data/text8'):
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(dest):
        print('Downloading text8 (~100 MB)...')
        urllib.request.urlretrieve(TEXT8_URL, 'data/text8.zip')
        with zipfile.ZipFile('data/text8.zip') as zf:
            zf.extractall('data')
        os.remove('data/text8.zip')
        print('Done.')
    return dest


def train(args):
    np.random.seed(args.seed)

    print('Loading corpus...')
    tokens = load_text(args.data)
    vocab, counts = build_vocab(tokens, min_count=args.min_count)
    V = len(vocab)
    idx2word = {i: w for w, i in vocab.items()}
    print(f'Vocab: {V} words  |  Corpus: {len(tokens)} tokens')

    tokens = subsample(tokens, vocab, counts, t=args.subsample_t)
    print(f'After subsampling: {len(tokens)} tokens')

    ns_dist = noise_distribution(vocab, counts)
    noise_table = build_noise_table(ns_dist)
    model = SkipGramNS(V, args.embed_dim)

    # build pairs once to estimate total steps for the LR schedule
    # (pairs are re-generated each epoch due to random window, count varies slightly)
    print('Building training pairs...')
    first_pairs = build_pairs(tokens, vocab, window=args.window)
    total_pairs_est = len(first_pairs) * args.epochs

    total_steps = 0
    for epoch in range(1, args.epochs + 1):
        pairs = first_pairs if epoch == 1 else build_pairs(tokens, vocab, window=args.window)
        np.random.shuffle(pairs)
        n = len(pairs)

        epoch_loss = 0.0
        t0 = time.time()

        for step, (center, ctx) in enumerate(pairs):
            # linear LR decay over full training run
            lr = max(args.min_lr, args.lr * (1.0 - total_steps / total_pairs_est))

            neg = noise_table[np.random.randint(0, len(noise_table), size=args.n_neg)]

            loss = model.forward(center, ctx, neg)
            model.backward(lr)

            epoch_loss += loss
            total_steps += 1

            if step > 0 and step % 500_000 == 0:
                avg = epoch_loss / step
                elapsed = time.time() - t0
                pct = 100 * step / n
                print(f'  [{epoch}/{args.epochs}] {pct:.0f}%  loss={avg:.4f}  lr={lr:.5f}  ({elapsed:.0f}s)')

        avg_loss = epoch_loss / n
        print(f'Epoch {epoch}/{args.epochs} - avg loss: {avg_loss:.4f}  ({time.time()-t0:.0f}s)')

    return model, vocab, idx2word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        default=None,   help='path to text corpus')
    parser.add_argument('--embed_dim',   type=int,   default=100)
    parser.add_argument('--window',      type=int,   default=5)
    parser.add_argument('--n_neg',       type=int,   default=5,   help='negatives per sample')
    parser.add_argument('--epochs',      type=int,   default=3)
    parser.add_argument('--lr',          type=float, default=0.025)
    parser.add_argument('--min_lr',      type=float, default=1e-4)
    parser.add_argument('--min_count',   type=int,   default=5)
    parser.add_argument('--subsample_t', type=float, default=1e-4)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--save',        default='vectors.npy')
    args = parser.parse_args()

    if args.data is None:
        args.data = download_text8()

    model, vocab, idx2word = train(args)

    np.save(args.save, model.W)
    vocab_path = args.save.replace('.npy', '_vocab.npy')
    np.save(vocab_path, vocab)
    print(f'\nSaved embeddings -> {args.save}')
    print(f'Saved vocab      -> {vocab_path}')


if __name__ == '__main__':
    main()