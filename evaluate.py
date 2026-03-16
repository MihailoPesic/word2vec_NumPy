"""
Quick evaluation of trained word vectors.

Usage:
    python evaluate.py vectors.npy
    python evaluate.py vectors.npy --word paris --analogy man king woman
"""

import argparse
import numpy as np


def load_vectors(path):
    W = np.load(path)
    vocab_path = path.replace('.npy', '_vocab.npy')
    vocab = np.load(vocab_path, allow_pickle=True).item()
    idx2word = {i: w for w, i in vocab.items()}
    return W, vocab, idx2word


def most_similar(word, W, vocab, idx2word, topn=10):
    if word not in vocab:
        print(f'"{word}" not in vocabulary')
        return []

    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
    W_norm = W / norms

    q = W_norm[vocab[word]]
    scores = W_norm @ q
    scores[vocab[word]] = -1  # exclude the query word itself

    top = np.argsort(-scores)[:topn]
    return [(idx2word[i], float(scores[i])) for i in top]


def analogy(a, b, c, W, vocab, idx2word, topn=5):
    """Solve  a : b :: c : ?   Finds the word closest to v_b - v_a + v_c."""
    missing = [w for w in (a, b, c) if w not in vocab]
    if missing:
        print(f'Words not in vocab: {missing}')
        return []

    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
    W_norm = W / norms

    def unit(w):
        return W_norm[vocab[w]]

    target = unit(b) - unit(a) + unit(c)
    target /= np.linalg.norm(target) + 1e-10

    scores = W_norm @ target
    exclude = {vocab[w] for w in (a, b, c)}

    results = []
    for i in np.argsort(-scores):
        if i not in exclude:
            results.append((idx2word[i], float(scores[i])))
        if len(results) == topn:
            break
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vectors', help='path to vectors.npy')
    parser.add_argument('--word',    default='king',          help='word for similarity query')
    parser.add_argument('--analogy', nargs=3, metavar=('A', 'B', 'C'),
                        default=['man', 'king', 'woman'],
                        help='solve A:B :: C:?')
    args = parser.parse_args()

    W, vocab, idx2word = load_vectors(args.vectors)
    print(f'Loaded {W.shape[0]} vectors of dim {W.shape[1]}')

    print(f'\nMost similar to "{args.word}":')
    for w, s in most_similar(args.word, W, vocab, idx2word):
        print(f'  {w:<20} {s:.4f}')

    a, b, c = args.analogy
    print(f'\nAnalogy  {a} : {b} :: {c} : ?')
    for w, s in analogy(a, b, c, W, vocab, idx2word):
        print(f'  {w:<20} {s:.4f}')


if __name__ == '__main__':
    main()