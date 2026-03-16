import re
import numpy as np
from collections import Counter


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return re.findall(r'[a-z]+', text)


def build_vocab(tokens, min_count=5):
    counts = Counter(tokens)
    vocab = {
        w: i for i, (w, c) in enumerate(counts.most_common())
        if c >= min_count
    }
    return vocab, counts


def subsample(tokens, vocab, counts, t=1e-4):
    """Discard frequent words probabilistically (Mikolov).

    Keep word w with probability (sqrt(f/t) + 1) * (t/f) where f is its
    corpus frequency. This is the formula from the original C code — slightly
    different from eq.5 in the paper but what's actually running there.
    """
    total = sum(counts[w] for w in vocab)
    keep = []
    for w in tokens:
        if w not in vocab:
            continue
        f = counts[w] / total
        p_keep = (np.sqrt(f / t) + 1) * (t / f)
        if p_keep >= 1.0 or np.random.random() < p_keep:
            keep.append(w)
    return keep


def build_pairs(tokens, vocab, window=5):
    """Generate (center, context) index pairs.

    Window size is drawn randomly from [1, window] per word — closer words
    end up as context targets more often on average.
    Note: this is a plain Python loop; for very large corpora it's the
    bottleneck, but vectorizing it cleanly is awkward.
    """
    ids = [vocab[w] for w in tokens if w in vocab]
    pairs = []
    for i, center in enumerate(ids):
        w = np.random.randint(1, window + 1)
        lo, hi = max(0, i - w), min(len(ids), i + w + 1)
        for j in range(lo, hi):
            if j != i:
                pairs.append((center, ids[j]))
    return np.array(pairs, dtype=np.int32)


def noise_distribution(vocab, counts, exp=0.75):
    """Unigram^0.75 distribution for negative sampling.

    Smoothing with 0.75 gives rare words a better chance of being sampled
    as negatives relative to their raw frequency.
    """
    freqs = np.array([counts[w] for w, _ in sorted(vocab.items(), key=lambda x: x[1])])
    freqs = freqs.astype(np.float64) ** exp
    return freqs / freqs.sum()


def build_noise_table(ns_dist, size=10_000_000):
    # np.random.choice with a custom p= recomputes the CDF on every call which
    # gets expensive in a tight training loop; precompute a table once and just
    # pick random indices into it — same trick used in the original C code
    return np.random.choice(len(ns_dist), size=size, p=ns_dist).astype(np.int32)