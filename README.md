# word2vec from scratch

Skip-gram word2vec with negative sampling, pure NumPy.

## How it works

Given a center word, the model tries to predict surrounding context words. Instead of a full softmax over the vocabulary (O(V) per step), negative sampling is used - for each real (center, context) pair, k random "noise" words are drawn and the task becomes binary classification: real context or noise?

Loss per sample:
```
L = -log σ(v_c · u_w)  -  Σ log σ(-v_n · u_w)
```

Two embedding matrices: `W` for center words, `C` for context. Only `W` is saved after training.

Other details:
- subsampling: frequent words (the, of, a...) are dropped with probability proportional to their frequency - helps both training speed and embedding quality
- noise distribution: unigram^0.75 instead of raw unigram, which gives rare words a better shot at being sampled as negatives
- dynamic window: window size is re-sampled from [1, max_window] per word, so closer context words appear in more training pairs on average

## Dataset

[text8](http://mattmahoney.net/dc/text8.zip) — ~100MB of cleaned Wikipedia. Downloads automatically on first run. Training 3 epochs takes a few hours on CPU with this pure Python/NumPy implementation.

## Usage

```bash
pip install numpy

python train.py                          # downloads text8, trains with defaults
python train.py --data myfile.txt        # custom corpus
python train.py --epochs 5 --embed_dim 200 --n_neg 10

python evaluate.py vectors.npy --word france
python evaluate.py vectors.npy --analogy man king woman
```

Main arguments: `--embed_dim` (100), `--window` (5), `--n_neg` (5), `--epochs` (3), `--lr` (0.025), `--min_count` (5).

## Files

```
preprocess.py   tokenization, vocab, subsampling, pair generation
word2vec.py     model - forward pass, loss, gradients, parameter update
train.py        training loop
evaluate.py     similarity and analogy queries
```