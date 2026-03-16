import numpy as np


def sigmoid(x):
    # clip to avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class SkipGramNS:
    """
    Skip-gram word2vec with negative sampling.

    Two matrices: W (V x d) for center words, C (V x d) for context words.
    Loss for a single (center, context, k negatives) sample:

        L = -log σ(v_c · u_w)  -  Σ log σ(-v_n · u_w)

    After training W is used as word representations; C is discarded
    (though averaging W and C can sometimes help a bit).
    """

    def __init__(self, vocab_size, embed_dim):
        V, d = vocab_size, embed_dim
        # W: small uniform init; C: zeros — same as original word2vec C code
        self.W = (np.random.rand(V, d) - 0.5) / d
        self.C = np.zeros((V, d))
        self._cache = None

    def forward(self, center, pos_ctx, neg_ctx):
        u     = self.W[center]      # center word vector
        v_pos = self.C[pos_ctx]     # positive context vector
        V_neg = self.C[neg_ctx]     # negative vectors, (k, d)

        pos_score = np.dot(u, v_pos)
        neg_scores = V_neg @ u      # (k,)

        loss = -np.log(sigmoid(pos_score) + 1e-10) \
               - np.sum(np.log(sigmoid(-neg_scores) + 1e-10))

        self._cache = (center, pos_ctx, neg_ctx, u, v_pos, V_neg, pos_score, neg_scores)
        return loss

    def backward(self, lr):
        center, pos_ctx, neg_ctx, u, v_pos, V_neg, pos_score, neg_scores = self._cache

        # d/ds[-log σ(s)]  = σ(s) - 1   → negative: pushes center toward positive context
        # d/ds[-log σ(-s)] = σ(s)        → positive: pushes center away from negatives
        pos_err = sigmoid(pos_score) - 1.0
        neg_err = sigmoid(neg_scores)  # (k,)

        grad_u     = pos_err * v_pos + (neg_err[:, None] * V_neg).sum(axis=0)
        grad_v_pos = pos_err * u
        grad_V_neg = neg_err[:, None] * u[None, :]

        self.W[center]  -= lr * grad_u
        self.C[pos_ctx] -= lr * grad_v_pos
        self.C[neg_ctx] -= lr * grad_V_neg